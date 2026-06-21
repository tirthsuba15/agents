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

FEATURE_NAMES = ["r1", "opex_flag", "vix_level", "volume_ratio", "day_of_week", "days_to_opex"]
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
    return _cache["model"]


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
    model    = _load_model()
    vec      = np.array([[features_dict[f] for f in FEATURE_NAMES]], dtype=float)
    proba    = float(model.predict_proba(vec)[0][1])   # P(momentum held)
    direction  = round((proba - 0.5) * 2, 4)
    conviction = round(abs(direction), 4)

    return {
        "model":      "momentum_xgb_v1",
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

    return {
        "r1":           r1,
        "opex_flag":    int(is_opex_week(today)),
        "vix_level":    vix,
        "volume_ratio": vol_ratio,
        "day_of_week":  today.weekday(),
        "days_to_opex": days_to_next_opex(today),
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
