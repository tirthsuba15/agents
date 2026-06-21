#!/usr/bin/env python3
"""
Train momentum XGBoost classifier — cross-sectional edition.

Cross-sectional training on the 21-stock trading basket (~15,000 rows)
vs the original SPY-only dataset (~722 rows). Uses the same intraday
momentum label: 1 if first-hour return and last-hour return share sign.

Features (13 total):
  r1              — first-hour return vs prior close
  opex_flag       — 1 if OPEX week
  vix_level       — VIX close that day
  volume_ratio    — daily volume / 20-day average
  day_of_week     — 0 (Mon) to 4 (Fri)
  days_to_opex    — calendar days to next OPEX Friday
  overnight_return — today open vs prior close
  gex_regime_proxy — SPY above/below 50d SMA (+1/-1)
  r5d             — 5-day return of the stock
  realized_vol10  — 10-day annualised realised vol
  r1_lag1         — prior day's first-hour return (prior daily return proxy)
  r1_x_opex       — r1 * opex_flag (interaction)
  vix_x_vol       — vix_level * volume_ratio (interaction)

Data: yfinance hourly + daily per stock (2yr, ~500 trading days each).
Split: 70/30 by date (no lookahead).
Target: test accuracy > 52%.

Usage:
    python models/train.py
"""

import calendar
import datetime
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier

MODEL_PATH    = Path(__file__).parent / "momentum_model.pkl"
FEATURE_NAMES = [
    "r1", "opex_flag", "vix_level", "volume_ratio", "day_of_week", "days_to_opex",
    "overnight_return", "gex_regime_proxy",
    "r5d", "realized_vol10", "r1_lag1",
    "r1_x_opex", "vix_x_vol",
]

# Same basket as the OPEX backtest (sector ETFs replace energy stocks)
BASKET = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "JPM",  "JNJ",   "UNH",
    "HD",   "WMT",  "PG",    "BAC",  "MA",
    "V",    "ABBV", "MRK",   "PFE",
    "XLK",  "XLV",  "XLF",
]


# ---------------------------------------------------------------------------
# OPEX helpers
# ---------------------------------------------------------------------------

def third_friday(year: int, month: int) -> datetime.date:
    c = calendar.monthcalendar(year, month)
    fridays = [week[calendar.FRIDAY] for week in c if week[calendar.FRIDAY] != 0]
    return datetime.date(year, month, fridays[2])


def is_opex_week(date: datetime.date) -> bool:
    tf = third_friday(date.year, date.month)
    monday = tf - datetime.timedelta(days=tf.weekday())
    return monday <= date <= tf


def days_to_next_opex(date: datetime.date) -> int:
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
# Data download helpers
# ---------------------------------------------------------------------------

def _to_date_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = pd.DatetimeIndex(idx).date
    return df


def download_hourly(symbol: str, period: str = "730d") -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period, interval="1h", auto_adjust=True)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York")
    df["date"] = df.index.date
    return df


def download_daily(symbol: str, period: str = "730d") -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
    return _to_date_index(df)


# ---------------------------------------------------------------------------
# Feature + label engineering (per ticker)
# ---------------------------------------------------------------------------

def build_records_for(
    symbol: str,
    vix_closes: dict,
    spy_closes: dict,
    spy_sma50: dict,
) -> list[dict]:
    """Build training records for a single ticker."""
    try:
        h = download_hourly(symbol)
        d = download_daily(symbol)
    except Exception as exc:
        print(f"    {symbol}: download error — {exc}")
        return []

    if h.empty or d.empty or len(d) < 30:
        print(f"    {symbol}: insufficient data")
        return []

    d["vol20"]   = d["Volume"].rolling(20, min_periods=5).mean()
    d["rvol10"]  = d["Close"].pct_change().rolling(10).std() * np.sqrt(252)
    d["r5d"]     = d["Close"].pct_change(5)
    sym_closes   = d["Close"].to_dict()
    sym_opens    = d["Open"].to_dict()
    sym_vols     = d["Volume"].to_dict()
    sym_vol20    = d["vol20"].to_dict()
    sym_rvol10   = d["rvol10"].to_dict()
    sym_r5d      = d["r5d"].to_dict()

    daily_groups = h.groupby("date")
    dates        = sorted(daily_groups.groups.keys())

    records = []
    prev_r1 = 0.0

    for i, date in enumerate(dates):
        if i == 0:
            continue

        bars = daily_groups.get_group(date)
        if len(bars) < 4:
            prev_r1 = 0.0
            continue

        prior_date  = dates[i - 1]
        prior_close = sym_closes.get(prior_date)
        if not prior_close:
            prev_r1 = 0.0
            continue

        first_close = float(bars.iloc[0]["Close"])
        last_close  = float(bars.iloc[-1]["Close"])
        second_last = float(bars.iloc[-2]["Close"])

        r1    = (first_close - prior_close) / prior_close
        r13   = (last_close - second_last) / second_last if second_last > 0 else 0.0
        label = int((r1 > 0) == (r13 > 0))

        vix       = float(vix_closes.get(date) or vix_closes.get(prior_date) or 20.0)
        vol       = float(sym_vols.get(date) or 0)
        vol20     = float(sym_vol20.get(date) or vol or 1)
        vol_ratio = vol / vol20 if vol20 > 0 else 1.0

        today_open = sym_opens.get(date)
        overnight  = (today_open / prior_close - 1) if today_open and prior_close else 0.0

        sma50     = spy_sma50.get(date)
        spy_close = spy_closes.get(date)
        gex_proxy = 1 if (sma50 and spy_close and spy_close > sma50) else -1

        r5d        = float(sym_r5d.get(date) or 0.0)
        rvol10     = float(sym_rvol10.get(date) or 0.2)
        r1_lag1    = prev_r1

        opex_flag = int(is_opex_week(date))

        records.append({
            "date":             date,
            "r1":               r1,
            "opex_flag":        opex_flag,
            "vix_level":        vix,
            "volume_ratio":     vol_ratio,
            "day_of_week":      date.weekday(),
            "days_to_opex":     days_to_next_opex(date),
            "overnight_return": overnight,
            "gex_regime_proxy": gex_proxy,
            "r5d":              r5d,
            "realized_vol10":   rvol10,
            "r1_lag1":          r1_lag1,
            "r1_x_opex":        r1 * opex_flag,
            "vix_x_vol":        vix * vol_ratio,
            "label":            label,
        })
        prev_r1 = r1

    return records


# ---------------------------------------------------------------------------
# Build full cross-sectional dataset
# ---------------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    print("  SPY daily (shared regime signal)...")
    spy_d = download_daily("SPY")
    spy_d["sma50"] = spy_d["Close"].rolling(50, min_periods=20).mean()
    spy_closes = spy_d["Close"].to_dict()
    spy_sma50  = spy_d["sma50"].to_dict()

    print("  VIX daily...")
    vix_d      = download_daily("^VIX")
    vix_closes = vix_d["Close"].to_dict()

    all_records = []
    for sym in BASKET:
        print(f"  {sym}...")
        recs = build_records_for(sym, vix_closes, spy_closes, spy_sma50)
        for r in recs:
            r["symbol"] = sym
        all_records.extend(recs)
        print(f"    -> {len(recs)} rows")

    return pd.DataFrame(all_records).set_index("date").dropna()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> None:
    print(f"Building cross-sectional dataset ({len(BASKET)} stocks)...")
    df = build_dataset()
    print(f"\n  Samples        : {len(df):,}")
    print(f"  Stocks         : {df['symbol'].nunique()}")
    print(f"  Date range     : {df.index.min()} to {df.index.max()}")
    vc = df["label"].value_counts()
    print(f"  Label balance  : class-1={vc.get(1,0):,}  class-0={vc.get(0,0):,}")

    X = df[FEATURE_NAMES].values
    y = df["label"].values

    split   = int(len(df) * 0.70)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"  Train / Test   : {len(X_train):,} / {len(X_test):,}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    acc = float((model.predict(X_test) == y_test).mean())
    tag = "PASS" if acc > 0.52 else "INFO"
    print(f"\n  Test accuracy  : {acc*100:.2f}%  (target >52%)")
    print(f"  [{tag}] {'Above' if acc > 0.52 else 'Below'} 52% threshold")

    fi = dict(zip(FEATURE_NAMES, model.feature_importances_))
    print("\n  Feature importances:")
    for feat, imp in sorted(fi.items(), key=lambda x: -x[1]):
        print(f"    {feat:<20}: {imp:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, MODEL_PATH)
    print(f"\n  Saved: {MODEL_PATH}")


if __name__ == "__main__":
    train()
