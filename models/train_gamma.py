#!/usr/bin/env python3
"""
Train gamma XGBoost classifier on price-based proxy features.
==============================================================
Free alternative to paid historical options data.
Universe : ~110 S&P 500 large-caps + sector ETFs
History  : 5 years of weekly data (~260 weeks per ticker)
Samples  : ~38,000 training rows (after NaN drop)

Proxy feature mapping (same names as live options features so the model
pkl loads transparently into gamma_model.py inference):

  iv_spread       <- HV30 - HV10              (IV term structure proxy)
  smirk           <- -skewness(20d returns)   (put demand proxy)
  pcr             <- downvol_fraction          (bearish flow proxy)
  gex_regime_flag <- sign(close - SMA20)       (trend / gamma regime proxy)
  vix_level       <- ^VIX daily close          (exact — free from yfinance)
  mom4w           <- 4-week price return        (short-term momentum)
  rsi14           <- 14-week RSI                (overbought / oversold)
  hpr52           <- close / 52wk-high          (breakout proximity)

Label: binary — 1 if stock outperformed SPY next week, 0 otherwise.

Usage:
    python models/train_gamma.py
"""

import datetime
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

MODEL_PATH    = Path(__file__).parent / "gamma_model.pkl"
FEATURE_NAMES = [
    "iv_spread", "smirk", "pcr", "gex_regime_flag", "vix_level",
    "mom4w", "rsi14", "hpr52",
]

START = "2019-01-01"
END   = datetime.date.today().isoformat()

# ---------------------------------------------------------------------------
# Universe — ~110 tickers across all S&P 500 sectors + sector ETFs
# ---------------------------------------------------------------------------
UNIVERSE = [
    # Mega-cap tech & semiconductors
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
    "ADBE", "ORCL", "CRM", "AMD", "NFLX", "CSCO", "QCOM",
    "AMAT", "KLAC", "LRCX", "SNPS", "CDNS", "PANW", "INTU", "AVGO",
    # Healthcare / pharma / biotech
    "UNH", "JNJ", "LLY", "ABT", "TMO", "SYK", "ISRG", "VRTX",
    "REGN", "GILD", "AMGN", "ZTS", "BSX", "BDX", "DHR", "CI", "MRK", "ABBV", "PFE",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "C",
    "SPGI", "MCO", "CB", "AON", "CME", "PGR",
    # Consumer discretionary & staples
    "WMT", "HD", "MCD", "SBUX", "COST", "TGT", "LOW", "TJX",
    "NKE", "BKNG", "PG", "KO", "PEP", "MDLZ", "CL", "MO",
    # Industrials
    "CAT", "HON", "UNP", "LMT", "RTX", "GE", "DE", "ETN", "ITW", "NSC", "ADP",
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB",
    # Utilities & real estate
    "NEE", "DUK", "SO", "PLD", "EQIX",
    # Communications
    "T", "VZ", "DIS",
    # Sector & broad ETFs (high option volume — great signal quality)
    "SPY", "QQQ", "IWM", "GLD", "TLT",
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB",
    "SMH", "HYG", "EFA", "EEM",
]
UNIVERSE = list(dict.fromkeys(UNIVERSE))


# ---------------------------------------------------------------------------
# Feature engineering from daily OHLCV
# ---------------------------------------------------------------------------

def hv(close: pd.Series, window: int) -> pd.Series:
    """Annualised historical volatility over `window` trading days."""
    return np.log(close / close.shift(1)).rolling(window).std() * np.sqrt(252)


def rsi_weekly(close_w: pd.Series, window: int = 14) -> pd.Series:
    """RSI applied to weekly close prices."""
    delta = close_w.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_weekly_features(daily: pd.DataFrame, vix_weekly: pd.Series) -> pd.DataFrame:
    """
    Compute weekly proxy features from daily OHLCV.
    Resamples to Friday-ending weeks (last trading day of each week).
    """
    close = daily["Close"]

    # iv_spread proxy: HV term structure (HV30 - HV10)
    iv_spread_d = hv(close, 30) - hv(close, 10)
    iv_spread_w = iv_spread_d.resample("W-FRI").last()

    # smirk proxy: negative of 20-day return skewness
    log_ret = np.log(close / close.shift(1))
    smirk_w = (-log_ret.rolling(20).skew()).resample("W-FRI").last()

    # pcr proxy: downvol / totalvol
    neg       = log_ret.copy().clip(upper=0)
    down_vol  = neg.rolling(20).std()
    total_vol = log_ret.rolling(20).std().replace(0, np.nan)
    pcr_w     = (down_vol / total_vol).resample("W-FRI").last()

    # gex_regime_flag proxy: sign(close - 20-day SMA)
    sma20 = close.rolling(20).mean()
    gex_w = np.sign(close - sma20).resample("W-FRI").last()

    # Price-factor features (weekly)
    close_w  = close.resample("W-FRI").last()
    mom4w_w  = close_w.pct_change(4)
    rsi14_w  = rsi_weekly(close_w, 14)
    hpr52_w  = close_w / close_w.rolling(52).max()

    df = pd.DataFrame({
        "iv_spread":       iv_spread_w,
        "smirk":           smirk_w,
        "pcr":             pcr_w,
        "gex_regime_flag": gex_w,
        "mom4w":           mom4w_w,
        "rsi14":           rsi14_w,
        "hpr52":           hpr52_w,
    })

    # Align VIX on the same weekly index
    df = df.join(vix_weekly.rename("vix_level"), how="left")
    df["vix_level"] = df["vix_level"].ffill()

    return df.replace([np.inf, -np.inf], np.nan).dropna()


def build_labels(close_daily: pd.Series, spy_close_daily: pd.Series) -> pd.Series:
    """
    Binary label: 1 if stock outperformed SPY next week, 0 otherwise.
    Shift(-1) aligns the NEXT week's return to the CURRENT week's features.
    """
    stock_w = close_daily.resample("W-FRI").last().pct_change().shift(-1)
    spy_w   = spy_close_daily.resample("W-FRI").last().pct_change().shift(-1)
    return ((stock_w - spy_w) > 0).astype(int).dropna()


# ---------------------------------------------------------------------------
# Build full dataset
# ---------------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    print(f"  Downloading SPY and VIX ({START} to {END})...")
    spy_daily = yf.Ticker("SPY").history(start=START, end=END, auto_adjust=True)
    spy_daily.index = spy_daily.index.tz_localize(None)
    spy_close = spy_daily["Close"]

    vix_daily = yf.Ticker("^VIX").history(start=START, end=END, auto_adjust=True)
    vix_daily.index = vix_daily.index.tz_localize(None)
    vix_weekly = vix_daily["Close"].resample("W-FRI").last()

    all_rows = []
    total = len(UNIVERSE)

    for i, sym in enumerate(UNIVERSE, 1):
        try:
            daily = yf.Ticker(sym).history(start=START, end=END, auto_adjust=True)
            if daily.empty or len(daily) < 100:
                print(f"  [{i:>3}/{total}] {sym:<6}: skipped (no data)")
                continue
            daily.index = daily.index.tz_localize(None)

            feats  = build_weekly_features(daily, vix_weekly)
            labels = build_labels(daily["Close"], spy_close)

            merged = feats.join(labels.rename("label"), how="inner").dropna()
            if len(merged) < 20:
                print(f"  [{i:>3}/{total}] {sym:<6}: skipped (<20 rows)")
                continue

            merged["symbol"] = sym
            all_rows.append(merged)
            print(f"  [{i:>3}/{total}] {sym:<6}: {len(merged):>4} weeks")

        except Exception as exc:
            print(f"  [{i:>3}/{total}] {sym:<6}: error — {exc}")

    if not all_rows:
        raise RuntimeError("No data built — check internet connection")

    return pd.concat(all_rows).rename_axis("date").reset_index()


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train() -> None:
    print(f"Building dataset — {len(UNIVERSE)} tickers, {START} to {END}...")
    df = build_dataset()

    vc = df["label"].value_counts()
    print(f"\n  Total rows      : {len(df):,}")
    print(f"  Tickers loaded  : {df['symbol'].nunique()}")
    print(f"  Date range      : {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Label balance   : class-1={vc.get(1,0):,}  class-0={vc.get(0,0):,}")

    # Date-ordered 70/30 split (no lookahead)
    df_sorted = df.sort_values("date")
    split     = int(len(df_sorted) * 0.70)
    train_df  = df_sorted.iloc[:split]
    test_df   = df_sorted.iloc[split:]

    X_train = train_df[FEATURE_NAMES].values
    y_train = train_df["label"].values
    X_test  = test_df[FEATURE_NAMES].values
    y_test  = test_df["label"].values

    print(f"\n  Train rows : {len(X_train):,}")
    print(f"  Test rows  : {len(X_test):,}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    preds = model.predict(X_test)
    acc   = float((preds == y_test).mean())

    tag = "PASS" if acc > 0.52 else "INFO"
    print(f"\n  Test accuracy        : {acc*100:.2f}%  (target >52%)")
    print(f"  [{tag}]")

    print("\n  Feature importances:")
    fi = dict(zip(FEATURE_NAMES, model.feature_importances_))
    for feat, imp in sorted(fi.items(), key=lambda x: -x[1]):
        print(f"    {feat:<20}: {imp:.4f}")

    joblib.dump({
        "model":         model,
        "feature_names": FEATURE_NAMES,
        "model_type":    "classifier",
        "dir_acc":       acc,
        "trained_on":    "price_proxies_v2",
        "universe_size": df["symbol"].nunique(),
        "train_date":    datetime.date.today().isoformat(),
    }, MODEL_PATH)
    print(f"\n  Saved: {MODEL_PATH}")


if __name__ == "__main__":
    train()
