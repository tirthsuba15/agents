#!/usr/bin/env python3
"""
Gamma Model -- Phase 4
=====================
Composite options-flow signal combining IV spread, volatility smirk,
put-call ratio, and GEX regime into a cross-sectional ranked score.

Formula (no ML needed for initial version):
  composite = 0.4 * (-pcr_rank) + 0.3 * iv_spread_rank + 0.3 * (-smirk_rank)
  where _rank = percentile rank across universe (0 to 1)

XGBoost regressor is trained IF a historical feature dataset is provided
(historical options snapshots are a paid data product). Falls back to the
weighted formula otherwise.

Features (per ticker):
  iv_spread      -- OI-weighted mean(call_IV - put_IV) across matched pairs
  smirk          -- IV(25-delta put) - IV(50-delta call)
  pcr            -- sum(put_volume) / sum(call_volume)
  gex_regime_flag-- +1 positive_gamma, -1 negative_gamma
  vix_level      -- current VIX
  macro_flag     -- 1 if current date is within 3 days of FOMC/CPI/NFP

Output: SignalObject with direction in [-1, +1] and conviction in [0, 1].

Usage (acceptance test):
  python models/gamma_model.py NVDA
  python models/gamma_model.py NVDA AAPL MSFT SPY   (custom universe)
"""

import calendar
import datetime
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import yfinance as yf
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.fred_client import get_risk_free_rate
from models.gex_engine import (
    fetch_options_chain,
    fetch_spot_and_div,
    compute_gex,
    find_zero_gamma_flip,
    get_gex_regime,
)

MIN_IV      = 0.01
MODEL_PATH  = Path(__file__).parent / "gamma_model.pkl"

# Feature names -- must match train_gamma.py FEATURE_NAMES
FEATURE_NAMES = [
    "iv_spread", "smirk", "pcr", "gex_regime_flag", "vix_level",
    "mom4w", "rsi14", "hpr52", "gex_x_vix", "macro_flag",
]

# Default cross-sectional universe used when running the acceptance test
DEFAULT_UNIVERSE = ["NVDA", "AAPL", "MSFT", "SPY"]

# Sector map for per-sector model routing (#29)
SECTOR_MAP: dict[str, str] = {}
for _sector, _tickers in {
    "tech": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "ADBE",
        "ORCL", "CRM", "AMD", "NFLX", "CSCO", "QCOM", "AMAT", "KLAC",
        "LRCX", "SNPS", "CDNS", "PANW", "INTU", "AVGO", "XLK", "SMH",
    ],
    "healthcare": [
        "UNH", "JNJ", "LLY", "ABT", "TMO", "SYK", "ISRG", "VRTX", "REGN",
        "GILD", "AMGN", "ZTS", "BSX", "BDX", "DHR", "CI", "MRK", "ABBV",
        "PFE", "XLV",
    ],
    "financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "C", "SPGI", "MCO",
        "CB", "AON", "CME", "PGR", "XLF", "HYG",
    ],
}.items():
    for _t in _tickers:
        SECTOR_MAP[_t] = _sector


# ---------------------------------------------------------------------------
# Time helper (local copy -- avoids importing private _tte)
# ---------------------------------------------------------------------------

def _tte(expiry: datetime.date) -> float:
    return max((expiry - datetime.date.today()).days / 365.25, 0.0)


# ---------------------------------------------------------------------------
# Black-Scholes delta (needed for smirk strike selection)
# ---------------------------------------------------------------------------

def _d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    return (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def bs_call_delta(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma < MIN_IV:
        return 0.0
    return math.exp(-q * T) * norm.cdf(_d1(S, K, T, r, q, sigma))


def bs_put_delta(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma < MIN_IV:
        return 0.0
    return math.exp(-q * T) * (norm.cdf(_d1(S, K, T, r, q, sigma)) - 1)


# ---------------------------------------------------------------------------
# Macro event gate flag (#31)
# ---------------------------------------------------------------------------

def is_macro_week(date=None) -> int:
    """
    Returns 1 if `date` (default: today) falls within 3 calendar days of a
    known high-impact macro event: FOMC decision, CPI release, or NFP.

    NFP:  first Friday of each month (day <= 7 and weekday == 4)
    CPI:  typically released 10th-16th of each month
    FOMC: hardcoded meeting end dates 2023-2026
    """
    if date is None:
        date = datetime.date.today()
    if hasattr(date, "date"):
        date = date.date()

    fomc_dates = {
        datetime.date(2023,2,1),  datetime.date(2023,3,22), datetime.date(2023,5,3),
        datetime.date(2023,6,14), datetime.date(2023,7,26), datetime.date(2023,9,20),
        datetime.date(2023,11,1), datetime.date(2023,12,13),
        datetime.date(2024,1,31), datetime.date(2024,3,20), datetime.date(2024,5,1),
        datetime.date(2024,6,12), datetime.date(2024,7,31), datetime.date(2024,9,18),
        datetime.date(2024,11,7), datetime.date(2024,12,18),
        datetime.date(2025,1,29), datetime.date(2025,3,19), datetime.date(2025,5,7),
        datetime.date(2025,6,18), datetime.date(2025,7,30), datetime.date(2025,9,17),
        datetime.date(2025,11,7), datetime.date(2025,12,10),
        datetime.date(2026,1,28), datetime.date(2026,3,18), datetime.date(2026,4,29),
        datetime.date(2026,6,17),
    }

    # Within 3 days of FOMC decision date
    for fd in fomc_dates:
        if abs((date - fd).days) <= 3:
            return 1

    # CPI week: 10th-16th of each month
    if 10 <= date.day <= 16:
        return 1

    return 0


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_iv_spread(chain: list) -> float:
    """
    OI-weighted mean of (call_IV - put_IV) across all matched (expiry, strike) pairs.
    Positive value = calls more expensive than puts (bullish skew).
    """
    num, denom = 0.0, 0.0
    for opt in chain:
        T = _tte(opt["expiry"])
        if T <= 0:
            continue
        civ, piv   = opt.get("call_iv", 0.0), opt.get("put_iv", 0.0)
        coi, poi   = opt.get("call_oi", 0),   opt.get("put_oi", 0)
        if civ >= MIN_IV and piv >= MIN_IV:
            weight  = float(coi + poi)
            num    += (civ - piv) * weight
            denom  += weight
    return num / denom if denom > 0 else 0.0


def compute_smirk(chain: list, spot: float, r: float, q: float) -> float:
    """
    Volatility smirk = IV(25-delta put) - IV(50-delta call).
    Finds the put option whose delta is closest to -0.25 and the call
    whose delta is closest to +0.50, looking across all expiries.
    Positive smirk = put skew premium exists (fear is priced in).
    """
    put_candidates:  list[tuple] = []   # (|delta_diff|, iv)
    call_candidates: list[tuple] = []

    for opt in chain:
        T = _tte(opt["expiry"])
        if T <= 0:
            continue
        K = opt["strike"]

        piv = opt.get("put_iv", 0.0)
        if piv >= MIN_IV:
            d = bs_put_delta(spot, K, T, r, q, piv)
            put_candidates.append((abs(d - (-0.25)), piv))

        civ = opt.get("call_iv", 0.0)
        if civ >= MIN_IV:
            d = bs_call_delta(spot, K, T, r, q, civ)
            call_candidates.append((abs(d - 0.50), civ))

    if not put_candidates or not call_candidates:
        return 0.0

    _, put_iv  = min(put_candidates,  key=lambda x: x[0])
    _, call_iv = min(call_candidates, key=lambda x: x[0])
    return put_iv - call_iv


def compute_pcr(chain: list) -> float:
    """
    Put-call ratio = sum(put_volume) / sum(call_volume).
    PCR > 1 = more puts traded (bearish sentiment).
    PCR < 1 = more calls traded (bullish sentiment).
    """
    total_put_vol  = sum(opt.get("put_volume",  0) for opt in chain)
    total_call_vol = sum(opt.get("call_volume", 0) for opt in chain)
    if total_call_vol == 0:
        return 1.0   # neutral default
    return total_put_vol / total_call_vol


def compute_gex_regime_flag(chain: list, spot: float, r: float, q: float) -> int:
    """Returns +1 for positive_gamma regime, -1 for negative_gamma."""
    net_gex = compute_gex(chain, spot, r, q)
    flip    = find_zero_gamma_flip(chain, spot, r, q)
    regime  = get_gex_regime(net_gex, spot, flip)
    return 1 if regime == "positive_gamma" else -1


def get_vix() -> float:
    try:
        bars = yf.Ticker("^VIX").history(period="2d", auto_adjust=True)
        return float(bars["Close"].iloc[-1]) if not bars.empty else 20.0
    except Exception:
        return 20.0


# ---------------------------------------------------------------------------
# Price-factor features (momentum, RSI, 52wk high proximity)
# ---------------------------------------------------------------------------

def _get_price_factors(symbol: str) -> dict:
    """Compute mom4w, rsi14, hpr52, gex_x_vix from the last ~400 days of daily closes."""
    defaults = {"mom4w": 0.0, "rsi14": 50.0, "hpr52": 1.0, "gex_x_vix": 0.0}
    try:
        bars  = yf.Ticker(symbol).history(period="400d", auto_adjust=True)
        if bars.empty or len(bars) < 30:
            return defaults
        close = bars["Close"]
        mom4w = float(close.iloc[-1] / close.iloc[-20] - 1) if len(close) >= 20 else 0.0
        delta = close.diff()
        gain  = float(delta.clip(lower=0).rolling(14).mean().iloc[-1])
        loss  = float((-delta.clip(upper=0)).rolling(14).mean().iloc[-1])
        rsi14 = 100 - (100 / (1 + gain / loss)) if loss > 0 else 100.0
        hpr52 = float(close.iloc[-1] / close.rolling(252).max().iloc[-1]) if len(close) >= 252 else 1.0
        # gex_x_vix: sign(close - SMA20) * HV10
        sma20     = float(close.rolling(20).mean().iloc[-1])
        hv10_val  = float(np.log(close / close.shift(1)).rolling(10).std().iloc[-1] * np.sqrt(252))
        gex_x_vix = float(np.sign(close.iloc[-1] - sma20)) * hv10_val
        return {"mom4w": mom4w, "rsi14": rsi14, "hpr52": hpr52, "gex_x_vix": gex_x_vix}
    except Exception:
        return defaults


# ---------------------------------------------------------------------------
# Full feature dict for one ticker
# ---------------------------------------------------------------------------

def compute_gamma_features(
    symbol: str,
    r: float,
    vix: float,
) -> dict | None:
    """
    Returns dict with all features, or None if options chain is unavailable.
    """
    try:
        spot, q = fetch_spot_and_div(symbol)
        if spot <= 0:
            return None
        chain = fetch_options_chain(symbol)
        if not chain:
            return None

        iv_spread       = compute_iv_spread(chain)
        smirk           = compute_smirk(chain, spot, r, q)
        pcr             = compute_pcr(chain)
        gex_regime_flag = compute_gex_regime_flag(chain, spot, r, q)
        price_factors   = _get_price_factors(symbol)
        macro_flag      = is_macro_week(datetime.date.today())

        return {
            "symbol":          symbol,
            "spot":            spot,
            "iv_spread":       iv_spread,
            "smirk":           smirk,
            "pcr":             pcr,
            "gex_regime_flag": gex_regime_flag,
            "vix_level":       vix,
            "macro_flag":      macro_flag,
            **price_factors,
        }
    except Exception as exc:
        print(f"    {symbol}: skipped ({exc})")
        return None


# ---------------------------------------------------------------------------
# Cross-sectional ranking and composite score
# ---------------------------------------------------------------------------

def _rank01(values: np.ndarray) -> np.ndarray:
    """Percentile rank normalised to [0, 1]. Ties use average rank."""
    if len(values) == 1:
        return np.array([0.5])
    r = rankdata(values)          # 1 to n
    return (r - 1) / (len(r) - 1)  # 0 to 1


def compute_composite_scores(feature_rows: list[dict]) -> list[dict]:
    """
    composite = 0.4 * (-pcr_rank) + 0.3 * iv_spread_rank + 0.3 * (-smirk_rank)

    Then normalise composite itself across the universe to [-1, +1] via
    min-max centred at 0: direction = 2*(score - min)/(max - min) - 1
    """
    iv_spreads = np.array([f["iv_spread"] for f in feature_rows])
    smirks     = np.array([f["smirk"]     for f in feature_rows])
    pcrs       = np.array([f["pcr"]       for f in feature_rows])

    iv_rank  = _rank01(iv_spreads)
    smirk_rk = _rank01(smirks)
    pcr_rk   = _rank01(pcrs)

    raw = 0.4 * (-pcr_rk) + 0.3 * iv_rank + 0.3 * (-smirk_rk)

    # Normalise to [-1, +1]
    lo, hi = raw.min(), raw.max()
    if hi > lo:
        direction = 2 * (raw - lo) / (hi - lo) - 1
    else:
        direction = np.zeros_like(raw)

    results = []
    for i, feat in enumerate(feature_rows):
        d = float(direction[i])
        results.append({
            **feat,
            "raw_composite":  float(raw[i]),
            "direction":      round(d, 4),
            "conviction":     round(abs(d), 4),
            "iv_spread_rank": round(float(iv_rank[i]),  4),
            "smirk_rank":     round(float(smirk_rk[i]), 4),
            "pcr_rank":       round(float(pcr_rk[i]),   4),
        })
    return results


# ---------------------------------------------------------------------------
# XGBoost path (future -- needs historical options snapshots)
# ---------------------------------------------------------------------------

def _try_load_xgb_model(target: str | None = None):
    """
    Load trained XGBoost model if it beats random (dir_acc > 0.51).
    Tries sector-specific model first (#29), falls back to full model.
    Returns (model, model_type, feature_names, normaliser, dir_acc) or all-None tuple.
    """
    import joblib

    candidates = []
    if target:
        sector = SECTOR_MAP.get(target, "other")
        sector_path = MODEL_PATH.parent / f"gamma_model_{sector}.pkl"
        if sector_path.exists():
            candidates.append(sector_path)
    candidates.append(MODEL_PATH)

    for path in candidates:
        if not path.exists():
            continue
        payload = joblib.load(path)
        dir_acc = payload.get("dir_acc", 0)
        if dir_acc < 0.51:
            continue
        return (
            payload.get("model"),
            payload.get("model_type", "regressor"),
            payload.get("feature_names", FEATURE_NAMES),
            payload.get("normaliser", 1.0),
            dir_acc,
        )
    return None, None, None, None, 0.0


def train_gamma_model(historical_data: list[dict]) -> None:
    """
    Train XGBoost regressor on historical weekly options features.

    historical_data: list of dicts, each with keys:
        iv_spread, smirk, pcr, gex_regime_flag, vix_level, label
    where label = +1 if stock outperformed SPY that week, -1 otherwise.

    This function is a stub -- it requires historical options snapshots
    (paid data product). See OPTIMIZATIONS.md item #7.
    """
    if len(historical_data) < 50:
        print("Insufficient historical data for XGBoost training.")
        print("Need weekly options snapshots -- see OPTIMIZATIONS.md item #7.")
        print("Using formula-based signal instead.")
        return

    import joblib
    import pandas as pd
    from xgboost import XGBRegressor

    FEAT = ["iv_spread", "smirk", "pcr", "gex_regime_flag", "vix_level"]
    df   = pd.DataFrame(historical_data).dropna()
    X    = df[FEAT].values
    y    = df["label"].values.astype(float)

    split  = int(len(df) * 0.70)
    model  = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                          random_state=42, n_jobs=-1)
    model.fit(X[:split], y[:split], eval_set=[(X[split:], y[split:])], verbose=False)

    corr = float(np.corrcoef(model.predict(X[split:]), y[split:])[0, 1])
    print(f"  IC (Pearson correlation): {corr:.4f}")

    joblib.dump({"model": model, "feature_names": FEAT}, MODEL_PATH)
    print(f"  Saved: {MODEL_PATH}")


# ---------------------------------------------------------------------------
# Main predict function
# ---------------------------------------------------------------------------

def predict_gamma(target: str, universe: list[str] | None = None) -> dict:
    """
    Compute gamma-flow signal for `target` ranked within `universe`.
    Falls back to formula (XGBoost used only if pkl exists and dir_acc > 0.51).
    When XGBoost is active, blends it 50/50 with the formula signal (#31).
    """
    if universe is None:
        universe = list(dict.fromkeys([target] + DEFAULT_UNIVERSE))
    elif target not in universe:
        universe = [target] + list(universe)

    r   = get_risk_free_rate()
    vix = get_vix()

    print(f"  Risk-free rate : {r*100:.3f}%  |  VIX : {vix:.1f}")
    print(f"  Universe       : {universe}")
    print(f"  Computing features...")

    feature_rows = []
    for sym in universe:
        print(f"    {sym}...", end=" ", flush=True)
        feat = compute_gamma_features(sym, r, vix)
        if feat:
            feature_rows.append(feat)
            print(f"iv_spread={feat['iv_spread']:+.4f}  "
                  f"smirk={feat['smirk']:+.4f}  "
                  f"pcr={feat['pcr']:.3f}  "
                  f"gex={'POS' if feat['gex_regime_flag']==1 else 'NEG'}  "
                  f"macro_flag={feat['macro_flag']}")
        else:
            print("no data")

    if not feature_rows:
        raise RuntimeError("No feature data available for any symbol in universe")

    target_row = next((f for f in feature_rows if f["symbol"] == target), None)
    if target_row is None:
        raise RuntimeError(f"{target} options chain unavailable")

    # Always compute formula score (needed for blend #31)
    scored = compute_composite_scores(feature_rows)
    formula_result    = next(s for s in scored if s["symbol"] == target)
    formula_direction = formula_result["direction"]

    # Try sector-specific XGBoost, fall back to full model (#29)
    xgb, model_type, feat_names, normaliser, dir_acc = _try_load_xgb_model(target)
    if xgb and model_type == "classifier" and dir_acc > 0.51:
        vec = np.array([[target_row.get(f, 0.0) for f in feat_names]])
        proba         = float(xgb.predict_proba(vec)[0][1])
        xgb_direction = round((proba - 0.5) * 2, 4)

        # Blend XGBoost + formula 50/50 (#31)
        blended   = 0.5 * xgb_direction + 0.5 * formula_direction
        direction  = round(float(np.clip(blended, -1, 1)), 4)
        conviction = round(abs(direction), 4)
        method     = "xgboost_blend_v1"
    elif xgb and model_type != "classifier":
        # Regressor fallback (legacy)
        vec           = np.array([[target_row.get(f, 0.0) for f in feat_names]])
        raw           = float(xgb.predict(vec)[0])
        xgb_direction = round(float(np.clip(raw / normaliser, -1, 1)), 4)
        blended   = 0.5 * xgb_direction + 0.5 * formula_direction
        direction  = round(float(np.clip(blended, -1, 1)), 4)
        conviction = round(abs(direction), 4)
        method     = "xgboost_blend_v1"
    else:
        direction  = formula_direction
        conviction = formula_result["conviction"]
        method     = "weighted_formula"

    return {
        "model":      f"gamma_{method}_v1",
        "timestamp":  datetime.datetime.now().isoformat(timespec="seconds"),
        "symbol":     target,
        "direction":  direction,
        "conviction": conviction,
        "signals": {
            "iv_spread":       round(target_row["iv_spread"],       6),
            "smirk":           round(target_row["smirk"],           6),
            "pcr":             round(target_row["pcr"],             4),
            "gex_regime":      "positive_gamma" if target_row["gex_regime_flag"] == 1
                               else "negative_gamma",
            "vix_level":       round(target_row["vix_level"],       2),
            "macro_flag":      target_row["macro_flag"],
        },
        "universe_size": len(feature_rows),
        "method":     method,
    }


# ---------------------------------------------------------------------------
# CLI -- acceptance test
# ---------------------------------------------------------------------------

def main() -> None:
    target   = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
    # Additional args after the target define a custom universe
    # If only the target is given, DEFAULT_UNIVERSE is used for cross-sectional ranking
    universe = [a.upper() for a in sys.argv[2:]] if len(sys.argv) > 2 else None

    print(f"\nGamma Model -- {target}")
    print("=" * 50)

    signal = predict_gamma(target, universe)

    print("\nSignalObject:")
    print(json.dumps(signal, indent=2))


if __name__ == "__main__":
    main()