#!/usr/bin/env python3
"""
basket_selection.py — Optuna basket optimization for Smart OPEX strategy
=========================================================================
Selects the best K-stock subset (K=3..15) from the full 21-stock basket
to maximize in-market annualized CAGR = mean_smart_opex_weekly_return * 52.

Train window : 2015-2020
OOS window   : 2021-2024

Fixed params:
  TREND_LOOKBACK  = 10
  MOMENTUM_FLOOR  = -0.01
  VIX_THRESHOLD   = 30
  MIN_TRADES      = 15

Usage:
    /Users/aarnavgutti/Documents/agents/venv/bin/python \
        /Users/aarnavgutti/Documents/agents/optimize/basket_selection.py
"""

import calendar
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yfinance as yf

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASKET = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "JPM",  "JNJ",   "UNH",
    "HD",   "WMT",  "PG",    "BAC",  "MA",
    "V",    "ABBV", "MRK",   "PFE",
    "XLK",  "XLV",  "XLF",
]

SPY            = "SPY"
TREND_LOOKBACK = 10
MOMENTUM_FLOOR = -0.01
VIX_THRESHOLD  = 30
MIN_TRADES     = 15
N_TRIALS       = 500
K_MIN, K_MAX   = 3, 15

TRAIN_START = "2015-01-01"
TRAIN_END   = "2020-12-31"
OOS_START   = "2021-01-01"
OOS_END     = "2024-12-31"
FULL_START  = "2015-01-01"
FULL_END    = "2024-12-31"

OUT_FILE = Path("/Users/aarnavgutti/Documents/agents/optimize/basket_selection_results.json")

# ---------------------------------------------------------------------------
# Gate dates (hardcoded FOMC/CPI/NFP 2015-2024)
# ---------------------------------------------------------------------------
_GATE_DATES: set = {datetime.date.fromisoformat(d) for d in [
    # FOMC
    "2015-01-28","2015-03-18","2015-04-29","2015-06-17","2015-07-29",
    "2015-09-17","2015-10-28","2015-12-16",
    "2016-01-27","2016-03-16","2016-04-27","2016-06-15","2016-07-27",
    "2016-09-21","2016-11-02","2016-12-14",
    "2017-02-01","2017-03-15","2017-05-03","2017-06-14","2017-07-26",
    "2017-09-20","2017-11-01","2017-12-13",
    "2018-01-31","2018-03-21","2018-05-02","2018-06-13","2018-08-01",
    "2018-09-26","2018-11-08","2018-12-19",
    "2019-01-30","2019-03-20","2019-05-01","2019-06-19","2019-07-31",
    "2019-09-18","2019-10-30","2019-12-11",
    "2020-01-29","2020-03-03","2020-03-15","2020-04-29","2020-06-10",
    "2020-07-29","2020-09-16","2020-11-05","2020-12-16",
    "2021-01-27","2021-03-17","2021-04-28","2021-06-16","2021-07-28",
    "2021-09-22","2021-11-03","2021-12-15",
    "2022-01-26","2022-03-16","2022-05-04","2022-06-15","2022-07-27",
    "2022-09-21","2022-11-02","2022-12-14",
    "2023-02-01","2023-03-22","2023-05-03","2023-06-14","2023-07-26",
    "2023-09-20","2023-11-01","2023-12-13",
    "2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31",
    "2024-09-18","2024-11-07","2024-12-18",
    # CPI
    "2015-01-16","2015-02-26","2015-03-24","2015-04-17","2015-05-22",
    "2015-06-18","2015-07-17","2015-08-19","2015-09-16","2015-10-15",
    "2015-11-17","2015-12-15",
    "2016-01-20","2016-02-19","2016-03-16","2016-04-14","2016-05-17",
    "2016-06-16","2016-07-15","2016-08-16","2016-09-16","2016-10-18",
    "2016-11-17","2016-12-15",
    "2017-01-18","2017-02-15","2017-03-15","2017-04-14","2017-05-12",
    "2017-06-14","2017-07-14","2017-08-11","2017-09-14","2017-10-13",
    "2017-11-15","2017-12-13",
    "2018-01-12","2018-02-14","2018-03-13","2018-04-11","2018-05-10",
    "2018-06-12","2018-07-12","2018-08-10","2018-09-13","2018-10-11",
    "2018-11-14","2018-12-12",
    "2019-01-11","2019-02-13","2019-03-12","2019-04-10","2019-05-10",
    "2019-06-12","2019-07-11","2019-08-13","2019-09-12","2019-10-10",
    "2019-11-13","2019-12-11",
    "2020-01-14","2020-02-13","2020-03-11","2020-04-10","2020-05-12",
    "2020-06-10","2020-07-14","2020-08-12","2020-09-11","2020-10-13",
    "2020-11-12","2020-12-10",
    "2021-01-13","2021-02-10","2021-03-10","2021-04-13","2021-05-12",
    "2021-06-10","2021-07-13","2021-08-11","2021-09-14","2021-10-13",
    "2021-11-10","2021-12-10",
    "2022-01-12","2022-02-10","2022-03-10","2022-04-12","2022-05-11",
    "2022-06-10","2022-07-13","2022-08-10","2022-09-13","2022-10-13",
    "2022-11-10","2022-12-13",
    "2023-01-12","2023-02-14","2023-03-14","2023-04-12","2023-05-10",
    "2023-06-13","2023-07-12","2023-08-10","2023-09-13","2023-10-12",
    "2023-11-14","2023-12-12",
    "2024-01-11","2024-02-13","2024-03-12","2024-04-10","2024-05-15",
    "2024-06-12","2024-07-11","2024-08-14","2024-09-11","2024-10-10",
    "2024-11-13","2024-12-11",
    # NFP
    "2015-01-09","2015-02-06","2015-03-06","2015-04-03","2015-05-08",
    "2015-06-05","2015-07-02","2015-08-07","2015-09-04","2015-10-02",
    "2015-11-06","2015-12-04",
    "2016-01-08","2016-02-05","2016-03-04","2016-04-01","2016-05-06",
    "2016-06-03","2016-07-08","2016-08-05","2016-09-02","2016-10-07",
    "2016-11-04","2016-12-02",
    "2017-01-06","2017-02-03","2017-03-10","2017-04-07","2017-05-05",
    "2017-06-02","2017-07-07","2017-08-04","2017-09-01","2017-10-06",
    "2017-11-03","2017-12-08",
    "2018-01-05","2018-02-02","2018-03-09","2018-04-06","2018-05-04",
    "2018-06-01","2018-07-06","2018-08-03","2018-09-07","2018-10-05",
    "2018-11-02","2018-12-07",
    "2019-01-04","2019-02-01","2019-03-08","2019-04-05","2019-05-03",
    "2019-06-07","2019-07-05","2019-08-02","2019-09-06","2019-10-04",
    "2019-11-01","2019-12-06",
    "2020-01-10","2020-02-07","2020-03-06","2020-04-03","2020-05-08",
    "2020-06-05","2020-07-02","2020-08-07","2020-09-04","2020-10-02",
    "2020-11-06","2020-12-04",
    "2021-01-08","2021-02-05","2021-03-05","2021-04-02","2021-05-07",
    "2021-06-04","2021-07-02","2021-08-06","2021-09-03","2021-10-08",
    "2021-11-05","2021-12-03",
    "2022-01-07","2022-02-04","2022-03-04","2022-04-01","2022-05-06",
    "2022-06-03","2022-07-08","2022-08-05","2022-09-02","2022-10-07",
    "2022-11-04","2022-12-02",
    "2023-01-06","2023-02-03","2023-03-10","2023-04-07","2023-05-05",
    "2023-06-02","2023-07-07","2023-08-04","2023-09-01","2023-10-06",
    "2023-11-03","2023-12-08",
    "2024-01-05","2024-02-02","2024-03-08","2024-04-05","2024-05-03",
    "2024-06-07","2024-07-05","2024-08-02","2024-09-06","2024-10-04",
    "2024-11-01","2024-12-06",
]}


# ---------------------------------------------------------------------------
# OPEX helpers
# ---------------------------------------------------------------------------
def third_friday(year: int, month: int) -> datetime.date:
    c = calendar.monthcalendar(year, month)
    fridays = [w[calendar.FRIDAY] for w in c if w[calendar.FRIDAY] != 0]
    return datetime.date(year, month, fridays[2])


def is_opex_week(date: datetime.date) -> bool:
    tf = third_friday(date.year, date.month)
    monday = tf - datetime.timedelta(days=tf.weekday())
    return monday <= date <= tf


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def fetch_bars(symbol: str) -> pd.DataFrame:
    ticker_sym = symbol if symbol != "^VIX" else "^VIX"
    df = yf.Ticker(ticker_sym).history(
        start=FULL_START, end=FULL_END, auto_adjust=True, actions=False
    )
    if df.empty:
        return pd.DataFrame()
    df.index = df.index.tz_localize(None).date if df.index.tz else df.index.date
    return df[["Open", "Close"]].rename(columns={"Open": "open", "Close": "close"})


def group_by_week(df: pd.DataFrame) -> dict:
    weeks: dict = {}
    for d in sorted(df.index):
        weeks.setdefault(d.isocalendar()[:2], []).append(d)
    return {k: weeks[k] for k in sorted(weeks)}


def weekly_returns_for(df: pd.DataFrame) -> pd.Series:
    opens  = df["open"].to_dict()
    closes = df["close"].to_dict()
    weeks  = group_by_week(df)
    out = {}
    for days in weeks.values():
        first, last = min(days), max(days)
        if first in opens and last in closes:
            out[first] = (closes[last] - opens[first]) / opens[first]
    return pd.Series(out, name="ret").sort_index()


# ---------------------------------------------------------------------------
# Build regime signals from SPY (simplified: MA trend + VIX + momentum)
# ---------------------------------------------------------------------------
def build_spy_signals(spy_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    spy_rets = weekly_returns_for(spy_df)

    # SPY weekly close (end-of-week)
    spy_closes = {}
    closes = spy_df["close"].to_dict()
    for days in group_by_week(spy_df).values():
        spy_closes[min(days)] = closes[max(days)]
    close_s = pd.Series(spy_closes).sort_index()

    ma = close_s.rolling(TREND_LOOKBACK, min_periods=1).mean().shift(1)
    prior_close = close_s.shift(1)
    uptrend = (prior_close > ma).reindex(spy_rets.index, fill_value=True)
    prior_ret = spy_rets.shift(1).reindex(spy_rets.index, fill_value=0.0)

    # VIX regime
    vix_low = pd.Series(True, index=spy_rets.index)
    if vix_df is not None and not vix_df.empty:
        vix_closes = {}
        vc = vix_df["close"].to_dict()
        for days in group_by_week(vix_df).values():
            vix_closes[min(days)] = vc[max(days)]
        vix_s = pd.Series(vix_closes).sort_index()
        vix_low = (vix_s < VIX_THRESHOLD).reindex(spy_rets.index).fillna(True).astype(bool)

    return pd.DataFrame({"uptrend": uptrend, "prior_ret": prior_ret, "vix_low": vix_low})


# ---------------------------------------------------------------------------
# Build week-level label table: is this week smart_opex?
# ---------------------------------------------------------------------------
def build_week_labels(spy_df: pd.DataFrame, spy_signals: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by week_start with columns:
      naive_opex, smart_opex
    This is regime-dependent but basket-independent — computed once.
    """
    spy_weeks = group_by_week(spy_df)
    week_days = {min(days): days for days in spy_weeks.values()}
    all_week_starts = sorted(week_days.keys())

    records = []
    for ws in all_week_starts:
        opex = is_opex_week(ws)
        days = week_days.get(ws, [ws])

        # Macro gate
        if opex and any(d in _GATE_DATES for d in days):
            opex = False

        uptrend   = bool(spy_signals.loc[ws, "uptrend"])   if ws in spy_signals.index else True
        prior_ret = float(spy_signals.loc[ws, "prior_ret"]) if ws in spy_signals.index else 0.0
        vix_low   = bool(spy_signals.loc[ws, "vix_low"])   if ws in spy_signals.index else True

        smart = opex and uptrend and prior_ret > MOMENTUM_FLOOR and vix_low
        records.append({"week_start": ws, "naive_opex": opex, "smart_opex": smart})

    return pd.DataFrame(records).set_index("week_start")


# ---------------------------------------------------------------------------
# Backtest for a given basket subset on a window
# ---------------------------------------------------------------------------
def backtest_subset(
    tickers: list,
    stock_weekly_rets: dict,
    week_labels: pd.DataFrame,
    start: datetime.date,
    end: datetime.date,
) -> dict:
    """
    Given:
      tickers          — list of included tickers
      stock_weekly_rets — {ticker: pd.Series(week_start -> ret)}
      week_labels      — DataFrame(week_start -> {smart_opex, naive_opex})
      start/end        — date window to evaluate

    Returns dict with keys: mean_ret, n_trades, cagr (annualized = mean*52)
    """
    if not tickers:
        return {"mean_ret": -999.0, "n_trades": 0, "cagr": -999.0}

    # Filter week_labels to the window
    mask = (week_labels.index >= start) & (week_labels.index <= end)
    wl = week_labels[mask]
    smart_weeks = wl[wl["smart_opex"]].index

    if len(smart_weeks) == 0:
        return {"mean_ret": 0.0, "n_trades": 0, "cagr": 0.0}

    # Build equal-weight basket return for each smart OPEX week
    weekly_basket_rets = []
    for ws in smart_weeks:
        stock_rets = []
        for t in tickers:
            s = stock_weekly_rets.get(t)
            if s is not None and ws in s.index:
                stock_rets.append(s[ws])
        if stock_rets:
            weekly_basket_rets.append(np.mean(stock_rets))

    if not weekly_basket_rets:
        return {"mean_ret": 0.0, "n_trades": 0, "cagr": 0.0}

    mean_ret = float(np.mean(weekly_basket_rets))
    n_trades = len(weekly_basket_rets)
    cagr = mean_ret * 52

    return {"mean_ret": mean_ret, "n_trades": n_trades, "cagr": cagr}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("OPEX Basket Selection via Optuna")
    print(f"Train: {TRAIN_START} - {TRAIN_END}   OOS: {OOS_START} - {OOS_END}")
    print(f"Trials: {N_TRIALS}   Min trades: {MIN_TRADES}   K: {K_MIN}..{K_MAX}")
    print("=" * 65)

    # --- Download all data once ---
    print("\n[1/3] Downloading data (one-time)...")
    spy_df = fetch_bars(SPY)
    print(f"  SPY: {len(spy_df)} bars")
    vix_df = fetch_bars("^VIX")
    print(f"  VIX: {len(vix_df)} bars")

    stock_weekly_rets: dict = {}
    for sym in BASKET:
        df = fetch_bars(sym)
        if df.empty or len(df) < 100:
            print(f"  {sym}: skipped (insufficient data)")
            continue
        stock_weekly_rets[sym] = weekly_returns_for(df)
        print(f"  {sym}: {len(df)} bars")

    available_tickers = list(stock_weekly_rets.keys())
    print(f"\n  Available tickers: {len(available_tickers)}")

    # --- Build regime labels once ---
    print("\n[2/3] Building regime signals...")
    spy_signals = build_spy_signals(spy_df, vix_df)
    week_labels = build_week_labels(spy_df, spy_signals)
    train_start = datetime.date.fromisoformat(TRAIN_START)
    train_end   = datetime.date.fromisoformat(TRAIN_END)
    oos_start   = datetime.date.fromisoformat(OOS_START)
    oos_end     = datetime.date.fromisoformat(OOS_END)

    n_smart_train = int(week_labels.loc[
        (week_labels.index >= train_start) & (week_labels.index <= train_end),
        "smart_opex"
    ].sum())
    n_smart_oos = int(week_labels.loc[
        (week_labels.index >= oos_start) & (week_labels.index <= oos_end),
        "smart_opex"
    ].sum())
    print(f"  Smart OPEX weeks in train: {n_smart_train}")
    print(f"  Smart OPEX weeks in OOS  : {n_smart_oos}")

    # --- Optuna optimization on TRAIN window ---
    print(f"\n[3/3] Running {N_TRIALS} Optuna trials on train window...")

    trial_log: list = []

    def objective(trial):
        # Suggest include/exclude for each ticker
        selected = [
            t for t in available_tickers
            if trial.suggest_categorical(f"include_{t}", [True, False])
        ]

        # Enforce K constraint
        k = len(selected)
        if k < K_MIN or k > K_MAX:
            return -999.0

        result = backtest_subset(
            selected, stock_weekly_rets, week_labels,
            train_start, train_end
        )

        if result["n_trades"] < MIN_TRADES:
            return -999.0

        trial_log.append({
            "tickers": selected,
            "k": k,
            "train_mean_ret_pct": result["mean_ret"] * 100,
            "train_cagr_pct": result["cagr"] * 100,
            "train_n_trades": result["n_trades"],
        })

        return result["cagr"]  # maximize

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\n  Optuna complete. Best train CAGR: {study.best_value * 100:.2f}%")

    # --- Collect and rank top 5 baskets from trial log ---
    if not trial_log:
        print("ERROR: No valid trials completed.")
        sys.exit(1)

    # Deduplicate by frozen set of tickers, keep best CAGR per set
    seen: dict = {}
    for entry in trial_log:
        key = frozenset(entry["tickers"])
        if key not in seen or entry["train_cagr_pct"] > seen[key]["train_cagr_pct"]:
            seen[key] = entry

    ranked = sorted(seen.values(), key=lambda x: x["train_cagr_pct"], reverse=True)

    # --- Evaluate top baskets on OOS ---
    print("\n--- Top 5 Baskets (Train) + OOS evaluation ---")
    top5_with_oos = []
    for i, entry in enumerate(ranked[:5]):
        oos_result = backtest_subset(
            entry["tickers"], stock_weekly_rets, week_labels,
            oos_start, oos_end
        )
        record = {
            "rank": i + 1,
            "tickers": sorted(entry["tickers"]),
            "k": entry["k"],
            "train_mean_ret_pct": round(entry["train_mean_ret_pct"], 4),
            "train_cagr_pct": round(entry["train_cagr_pct"], 2),
            "train_n_trades": entry["train_n_trades"],
            "oos_mean_ret_pct": round(oos_result["mean_ret"] * 100, 4),
            "oos_cagr_pct": round(oos_result["cagr"] * 100, 2),
            "oos_n_trades": oos_result["n_trades"],
        }
        top5_with_oos.append(record)
        print(
            f"  #{i+1}  K={entry['k']:2d}  "
            f"Train CAGR={entry['train_cagr_pct']:6.2f}%  "
            f"OOS CAGR={oos_result['cagr']*100:6.2f}%  "
            f"OOS trades={oos_result['n_trades']}  "
            f"Tickers={sorted(entry['tickers'])}"
        )

    winner = top5_with_oos[0]

    # --- Baseline: all stocks on OOS ---
    baseline_oos = backtest_subset(
        available_tickers, stock_weekly_rets, week_labels,
        oos_start, oos_end
    )
    print(f"\n  Baseline (all {len(available_tickers)} stocks) OOS CAGR: "
          f"{baseline_oos['cagr']*100:.2f}%  "
          f"(n_trades={baseline_oos['n_trades']})")

    print(f"\n  WINNER: {winner['tickers']}")
    print(f"    Train CAGR : {winner['train_cagr_pct']:.2f}%")
    print(f"    OOS CAGR   : {winner['oos_cagr_pct']:.2f}%")
    print(f"    OOS trades : {winner['oos_n_trades']}")

    # --- Save results ---
    output = {
        "meta": {
            "train_window": [TRAIN_START, TRAIN_END],
            "oos_window": [OOS_START, OOS_END],
            "n_trials": N_TRIALS,
            "k_range": [K_MIN, K_MAX],
            "min_trades": MIN_TRADES,
            "trend_lookback": TREND_LOOKBACK,
            "momentum_floor": MOMENTUM_FLOOR,
            "vix_threshold": VIX_THRESHOLD,
        },
        "baseline_oos": {
            "tickers": sorted(available_tickers),
            "oos_cagr_pct": round(baseline_oos["cagr"] * 100, 2),
            "oos_n_trades": baseline_oos["n_trades"],
        },
        "top5_baskets": top5_with_oos,
        "winner": winner,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {OUT_FILE}")
    print("\nDone.")

    return winner


if __name__ == "__main__":
    main()
