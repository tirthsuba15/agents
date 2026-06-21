#!/usr/bin/env python3
"""
Momentum model inference — Phase 3.

Loads models/momentum_model.pkl (trained by models/train.py) and
returns a SignalObject dict for a given feature set.

direction  = (P(class=1) - 0.5) * 2   maps [0,1] proba -> [-1, +1]
conviction = abs(direction)             [0, 1]

Regime from GEX engine feeds into conviction scaling (handled by Meta-agent).

Usage (acceptance test):
    python models/momentum_model.py
    python models/momentum_model.py AAPL
"""

import calendar
import datetime
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import yfinance as yf

FEATURE_NAMES = [
    "r1", "opex_flag", "vix_level", "volume_ratio", "day_of_week", "days_to_opex",
    "overnight_return", "gex_regime_proxy",
    "r5d", "realized_vol10", "r1_lag1",
    "r1_x_opex", "vix_x_vol",
    "rsi14", "beta_spy20", "macd_signal",   # new
]
MODEL_PATH    = Path(__file__).parent / "momentum_model.pkl"


# ---------------------------------------------------------------------------
# OPEX helpers (mirrors backtest/opex_backtest.py)
# ---------------------------------------------------------------------------

def third_friday(year: int, month: int) -> datetime.date:
    """Return the date of the third Friday of the given month."""
    c = calendar.monthcalendar(year, month)
    fridays = [week[calendar.FRIDAY] for week in c if week[calendar.FRIDAY] != 0]
    return datetime.date(year, month, fridays[2])


def is_opex_week(date: datetime.date) -> bool:
    """True if date falls Mon-Fri of OPEX week."""
    tf     = third_friday(date.year, date.month)
    monday = tf - datetime.timedelta(days=tf.weekday())
    return monday <= date <= tf


def days_to_next_opex(date: datetime.date) -> int:
    """Calendar days until next OPEX Friday. 0 if already in OPEX week."""
    if is_opex_week(date):
        return 0
    year, month = date.year, date.month
    tf = third_friday(year, month)
    if date < tf:
        return (tf - date).days
    month = month + 1 if month < 12 else 1
    year  = year if month > 1 else year + 1
    return (third_friday(year, month) - date).days


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

_cache: dict = {}


def _load_model():
    if "model" not in _cache:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No model at {MODEL_PATH}. Run:  python models/train.py"
            )
        payload = joblib.load(MODEL_PATH)
        _cache["model"] = payload["model"]
        _cache["lgbm"]  = payload.get("lgbm")
    return _cache["model"], _cache.get("lgbm")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_momentum(features_dict: dict) -> dict:
    """
    Input:  features_dict with keys matching FEATURE_NAMES
    Output: SignalObject dict

    direction  in [-1, +1]:  +1 = strong long, -1 = strong short
    conviction in [ 0,  1]:  distance from neutral (0.5 proba)
    """
    xgb_model, lgbm_model = _load_model()
    vec = np.array([[features_dict[f] for f in FEATURE_NAMES]], dtype=float)

    xgb_proba = float(xgb_model.predict_proba(vec)[0][1])

    if lgbm_model is not None:
        lgbm_proba = float(lgbm_model.predict_proba(vec)[0][1])
        proba = (xgb_proba + lgbm_proba) / 2
    else:
        proba = xgb_proba

    direction  = round((proba - 0.5) * 2, 4)
    conviction = round(abs(direction), 4)

    return {
        "model":      "momentum_ensemble_v2",
        "timestamp":  datetime.datetime.now().isoformat(timespec="seconds"),
        "direction":  direction,
        "conviction": conviction,
        "opex_flag":  int(features_dict["opex_flag"]),
        "r1":         round(float(features_dict["r1"]), 6),
        "features":   {k: round(float(features_dict[k]), 6) for k in FEATURE_NAMES},
    }


# ---------------------------------------------------------------------------
# Live feature builder
# ---------------------------------------------------------------------------

def get_live_features(symbol: str = "SPY") -> dict:
    """
    Compute all six features from live market data via yfinance.
    r1 uses the 30th 1-min bar (approx 10:00 AM) vs prior close.
    If intraday data is unavailable (pre-open / weekend), falls back to
    prior day's open-vs-prior-close return.
    """
    today  = datetime.date.today()
    ticker = yf.Ticker(symbol)

    # Daily bars for volume_ratio and prior close
    daily = ticker.history(period="30d", auto_adjust=True)
    if daily.index.tz is not None:
        daily.index = daily.index.tz_localize(None)
    daily.index = daily.index.date

    prior_close  = float(daily["Close"].iloc[-2]) if len(daily) >= 2 else float(daily["Close"].iloc[-1])
    today_vol    = float(daily["Volume"].iloc[-1])
    vol20        = float(daily["Volume"].rolling(20, min_periods=5).mean().iloc[-1])
    vol_ratio    = today_vol / vol20 if vol20 > 0 else 1.0

    # r1 — first-30-min return
    r1 = 0.0
    try:
        intraday = ticker.history(period="1d", interval="1m", auto_adjust=True)
        if not intraday.empty:
            if len(intraday) >= 30:
                # 30th bar is approximately 10:00 AM
                price_30m = float(intraday["Close"].iloc[29])
            else:
                price_30m = float(intraday["Close"].iloc[-1])
            r1 = (price_30m - prior_close) / prior_close
    except Exception:
        # Pre-open or no intraday data: use today's open as proxy
        if "Open" in daily.columns and len(daily) > 0:
            r1 = (float(daily["Open"].iloc[-1]) - prior_close) / prior_close

    # VIX
    vix = 20.0
    try:
        vix_bars = yf.Ticker("^VIX").history(period="2d", auto_adjust=True)
        if not vix_bars.empty:
            vix = float(vix_bars["Close"].iloc[-1])
    except Exception:
        pass

    # overnight_return: today's open vs prior close
    overnight = 0.0
    if "Open" in daily.columns and len(daily) > 0:
        today_open = float(daily["Open"].iloc[-1])
        overnight  = (today_open / prior_close - 1) if prior_close else 0.0

    # gex_regime_proxy: +1 if above 50-day SMA
    close_s    = daily["Close"]
    sma50      = float(close_s.rolling(50, min_periods=20).mean().iloc[-1])
    today_close = float(close_s.iloc[-1])
    gex_proxy  = 1 if today_close > sma50 else -1

    # r5d: 5-day return
    r5d = float(close_s.iloc[-1] / close_s.iloc[-6] - 1) if len(close_s) >= 6 else 0.0

    # realized_vol10: 10-day annualised volatility
    rvol10 = float(close_s.pct_change().rolling(10).std().iloc[-1] * np.sqrt(252))

    # r1_lag1: prior day's open-vs-prior-close return as proxy
    r1_lag1 = 0.0
    if len(daily) >= 3 and "Open" in daily.columns:
        r1_lag1 = float(daily["Open"].iloc[-2] / daily["Close"].iloc[-3] - 1)

    # RSI-14
    delta = close_s.pct_change()
    gain  = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs    = gain / loss.replace(0, 1e-9)
    rsi14 = float(100 - 100 / (1 + rs.iloc[-1]))
    if np.isnan(rsi14): rsi14 = 50.0

    # Beta vs SPY (20-day rolling)
    try:
        spy_hist = yf.Ticker("SPY").history(period="30d", auto_adjust=True)
        if spy_hist.index.tz is not None:
            spy_hist.index = spy_hist.index.tz_localize(None)
        spy_hist.index = spy_hist.index.date
        spy_rets = spy_hist["Close"].pct_change()
        stock_rets_full = close_s.pct_change()
        common = spy_rets.index.intersection(stock_rets_full.index)[-20:]
        if len(common) >= 10:
            cov = np.cov(stock_rets_full.reindex(common), spy_rets.reindex(common))[0][1]
            var = float(spy_rets.reindex(common).var())
            beta_spy20 = cov / var if var > 0 else 1.0
        else:
            beta_spy20 = 1.0
    except Exception:
        beta_spy20 = 1.0

    # MACD signal
    ema12 = float(close_s.ewm(span=12, adjust=False).mean().iloc[-1])
    ema26 = float(close_s.ewm(span=26, adjust=False).mean().iloc[-1])
    last_close = float(close_s.iloc[-1])
    macd_signal_val = (ema12 - ema26) / last_close if last_close > 0 else 0.0

    opex_flag = int(is_opex_week(today))

    return {
        "r1":               r1,
        "opex_flag":        opex_flag,
        "vix_level":        vix,
        "volume_ratio":     vol_ratio,
        "day_of_week":      today.weekday(),
        "days_to_opex":     days_to_next_opex(today),
        "overnight_return": overnight,
        "gex_regime_proxy": gex_proxy,
        "r5d":              r5d,
        "realized_vol10":   rvol10,
        "r1_lag1":          r1_lag1,
        "r1_x_opex":        r1 * opex_flag,
        "vix_x_vol":        vix * vol_ratio,
        "rsi14":       rsi14,
        "beta_spy20":  beta_spy20,
        "macd_signal": macd_signal_val,
    }


# ---------------------------------------------------------------------------
# CLI — acceptance test
# ---------------------------------------------------------------------------

def main() -> None:
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"

    print(f"Fetching live features for {symbol}...")
    features = get_live_features(symbol)
    print("Features:")
    for k, v in features.items():
        print(f"  {k:<18}: {v:.4f}")

    print("\nRunning inference...")
    signal = predict_momentum(features)
    signal["symbol"] = symbol

    print("\nSignalObject:")
    print(json.dumps(signal, indent=2))


if __name__ == "__main__":
    main()
