#!/usr/bin/env python3
"""
Train momentum XGBoost classifier — Gao et al. (2018) setup.

Labels: 1 if first-hour return (r1 proxy for 30-min) and last-hour return
        (r13 proxy) have the same sign — intraday momentum persisted.
        0 otherwise.

Features:
  r1            — first-hour return vs prior close
  opex_flag     — 1 if OPEX week
  vix_level     — VIX close that day
  volume_ratio  — daily volume / 20-day average
  day_of_week   — 0 (Mon) to 4 (Fri)
  days_to_opex  — calendar days to next OPEX Friday (0 if in OPEX week)

Data: yfinance hourly SPY (2 years, ~500 trading days).
Split: 70 / 30 by date (no lookahead).
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
FEATURE_NAMES = ["r1", "opex_flag", "vix_level", "volume_ratio", "day_of_week", "days_to_opex"]


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
    """Normalise a yfinance DataFrame index to plain date objects."""
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
# Feature + label engineering
# ---------------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    print("  SPY hourly (2 yr)...")
    spy_h = download_hourly("SPY")

    print("  SPY daily (2 yr)...")
    spy_d = download_daily("SPY")
    spy_d["vol20"] = spy_d["Volume"].rolling(20, min_periods=5).mean()

    print("  VIX daily (2 yr)...")
    vix_d = download_daily("^VIX")

    spy_closes  = spy_d["Close"].to_dict()
    spy_vols    = spy_d["Volume"].to_dict()
    spy_vol20   = spy_d["vol20"].to_dict()
    vix_closes  = vix_d["Close"].to_dict()

    daily_groups = spy_h.groupby("date")
    dates = sorted(daily_groups.groups.keys())

    records = []
    for i, date in enumerate(dates):
        if i == 0:
            continue

        bars = daily_groups.get_group(date)
        if len(bars) < 4:   # need at least 4 hourly bars for valid r13
            continue

        prior_date  = dates[i - 1]
        prior_close = spy_closes.get(prior_date)
        if not prior_close:
            continue

        first_close  = float(bars.iloc[0]["Close"])
        last_close   = float(bars.iloc[-1]["Close"])
        second_last  = float(bars.iloc[-2]["Close"])

        r1  = (first_close - prior_close) / prior_close
        r13 = (last_close - second_last) / second_last if second_last > 0 else 0.0
        label = int((r1 > 0) == (r13 > 0))   # 1 if momentum held, 0 if reversed

        vix      = float(vix_closes.get(date) or vix_closes.get(prior_date) or 20.0)
        vol      = float(spy_vols.get(date) or 0)
        vol20    = float(spy_vol20.get(date) or vol or 1)
        vol_ratio = vol / vol20 if vol20 > 0 else 1.0

        records.append({
            "date":         date,
            "r1":           r1,
            "opex_flag":    int(is_opex_week(date)),
            "vix_level":    vix,
            "volume_ratio": vol_ratio,
            "day_of_week":  date.weekday(),
            "days_to_opex": days_to_next_opex(date),
            "label":        label,
        })

    return pd.DataFrame(records).set_index("date").dropna()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> None:
    print("Building training dataset...")
    df = build_dataset()
    print(f"  Samples        : {len(df)}")
    print(f"  Date range     : {df.index[0]} to {df.index[-1]}")
    vc = df["label"].value_counts()
    print(f"  Label balance  : class-1={vc.get(1,0)}  class-0={vc.get(0,0)}")

    X = df[FEATURE_NAMES].values
    y = df["label"].values

    split   = int(len(df) * 0.70)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"  Train / Test   : {len(X_train)} / {len(X_test)}")

    model = XGBClassifier(
        n_estimators=200,
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
        print(f"    {feat:<18}: {imp:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, MODEL_PATH)
    print(f"\n  Saved: {MODEL_PATH}")


if __name__ == "__main__":
    train()
