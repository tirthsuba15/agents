#!/usr/bin/env python3
"""
OPEX CAGR Parameter Search — Optuna-based optimization
=======================================================
Objective: maximize in-market annualized CAGR on 8-stock high-performance basket.
Train window: 2015-2020 | OOS gate: 2021-2024

Usage:
    /Users/aarnavgutti/Documents/agents/venv/bin/python \
        /Users/aarnavgutti/Documents/agents/optimize/param_cagr_search.py
"""

import calendar
import datetime
import json
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASKET = ["NVDA", "JNJ", "UNH", "WMT", "HD", "ABBV", "AAPL", "V"]
SPY_SYM = "SPY"
VIX_SYM = "^VIX"

START_DATE = "2013-01-01"   # extra history for rolling MA warmup
END_DATE   = "2024-12-31"

TRAIN_START = datetime.date(2015, 1, 1)
TRAIN_END   = datetime.date(2020, 12, 31)
OOS_START   = datetime.date(2021, 1, 1)
OOS_END     = datetime.date(2024, 12, 31)

N_TRIALS = 600
MIN_TRADES = 15

OUT_PATH = Path("/Users/aarnavgutti/Documents/agents/optimize/param_cagr_results.json")

# ---------------------------------------------------------------------------
# Hardcoded FOMC / CPI / NFP gate dates (2015-2024)
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
# Helpers
# ---------------------------------------------------------------------------

def third_friday(year: int, month: int) -> datetime.date:
    c = calendar.monthcalendar(year, month)
    fridays = [week[calendar.FRIDAY] for week in c if week[calendar.FRIDAY] != 0]
    return datetime.date(year, month, fridays[2])


def is_opex_week(date: datetime.date) -> bool:
    tf = third_friday(date.year, date.month)
    monday = tf - datetime.timedelta(days=tf.weekday())
    return monday <= date <= tf


def fetch_daily(symbol: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=START_DATE, end=END_DATE, auto_adjust=True, actions=False)
    if df.empty:
        return pd.DataFrame()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index).date
    return df[["Open", "Close"]].rename(columns={"Open": "open", "Close": "close"})


def group_by_week(df: pd.DataFrame) -> dict:
    weeks: dict = {}
    for d in sorted(df.index):
        key = d.isocalendar()[:2]
        weeks.setdefault(key, []).append(d)
    return {k: weeks[k] for k in sorted(weeks)}


def weekly_returns(df: pd.DataFrame) -> pd.Series:
    """Compute Mon-open to Fri-close weekly returns."""
    opens  = df["open"].to_dict()
    closes = df["close"].to_dict()
    weeks  = group_by_week(df)
    out = {}
    for days in weeks.values():
        first, last = min(days), max(days)
        if first in opens and last in closes and opens[first] > 0:
            out[first] = (closes[last] - opens[first]) / opens[first]
    return pd.Series(out).sort_index()


def build_spy_weekly_close(spy_df: pd.DataFrame) -> pd.Series:
    closes = spy_df["close"].to_dict()
    weeks  = group_by_week(spy_df)
    out = {}
    for days in weeks.values():
        out[min(days)] = closes[max(days)]
    return pd.Series(out).sort_index()


# ---------------------------------------------------------------------------
# Data loading (ONCE — stored in module globals)
# ---------------------------------------------------------------------------

print("Downloading data once...")

print(f"  SPY...")
SPY_DF = fetch_daily(SPY_SYM)
print(f"  ^VIX...")
VIX_DF = fetch_daily(VIX_SYM)

print(f"  Basket: {BASKET}")
_basket_rets = {}
for sym in BASKET:
    df = fetch_daily(sym)
    if not df.empty:
        _basket_rets[sym] = weekly_returns(df)
        print(f"    {sym}: {len(df)} bars")

BASKET_WEEKLY = pd.DataFrame(_basket_rets).dropna(how="all").mean(axis=1)
BASKET_WEEKLY.name = "ret"

SPY_WEEKLY_CLOSE = build_spy_weekly_close(SPY_DF)
SPY_WEEKLY_RET   = weekly_returns(SPY_DF)

# VIX weekly close (last day of each week)
_vix_closes = VIX_DF["close"].to_dict()
_vix_weeks  = group_by_week(VIX_DF)
VIX_WEEKLY_CLOSE = pd.Series(
    {min(days): _vix_closes[max(days)] for days in _vix_weeks.values()}
).sort_index()

# Week-day lookup for blocked-date check
SPY_WEEK_DAYS: dict = {}
for _days in group_by_week(SPY_DF).values():
    SPY_WEEK_DAYS[min(_days)] = _days

print(f"\nData ready: {len(BASKET_WEEKLY)} basket weeks, {len(SPY_WEEKLY_CLOSE)} SPY weeks\n")

# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    trend_lookback: int,
    momentum_floor: float,
    vix_threshold: float,
    kelly_fraction: float,
    date_start: datetime.date,
    date_end: datetime.date,
) -> dict:
    """
    Run Smart OPEX strategy with given params on [date_start, date_end].
    Returns dict with n_trades, mean_ret, cagr (kelly-sized annualized).
    """
    # Build SPY rolling MA with the requested lookback
    spy_close = SPY_WEEKLY_CLOSE.copy()
    spy_ma    = spy_close.rolling(trend_lookback, min_periods=1).mean().shift(1)
    spy_prior = spy_close.shift(1)
    spy_uptrend = (spy_prior > spy_ma)

    # Prior week basket return
    basket_prior_ret = BASKET_WEEKLY.shift(1)

    # Collect active weeks
    rets = []
    all_weeks = [w for w in BASKET_WEEKLY.index if date_start <= w <= date_end]

    for ws in all_weeks:
        if not is_opex_week(ws):
            continue

        # Macro gate
        days = SPY_WEEK_DAYS.get(ws, [ws])
        if any(d in _GATE_DATES for d in days):
            continue

        # Regime: uptrend
        uptrend = bool(spy_uptrend.get(ws, True))
        if not uptrend:
            continue

        # Momentum floor (prior week basket return)
        prior_ret = float(basket_prior_ret.get(ws, 0.0))
        if prior_ret <= momentum_floor:
            continue

        # VIX filter
        vix_val = float(VIX_WEEKLY_CLOSE.get(ws, 0.0))
        if vix_val > vix_threshold and vix_val > 0:
            continue

        # Basket return this week
        if ws not in BASKET_WEEKLY.index:
            continue
        week_ret = float(BASKET_WEEKLY[ws])

        rets.append(week_ret)

    n = len(rets)
    if n == 0:
        return {"n_trades": 0, "mean_ret": 0.0, "cagr": 0.0}

    mean_ret = float(np.mean(rets))
    # CAGR = mean weekly return × kelly_fraction × 52
    cagr = mean_ret * kelly_fraction * 52.0

    return {"n_trades": n, "mean_ret": mean_ret, "cagr": cagr}


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial) -> float:
    trend_lookback  = trial.suggest_int("trend_lookback",   4, 52)
    momentum_floor  = trial.suggest_float("momentum_floor", -0.05, 0.01)
    vix_threshold   = trial.suggest_float("vix_threshold",  15.0, 45.0)
    kelly_fraction  = trial.suggest_float("kelly_fraction",  0.25, 1.5)

    result = evaluate(
        trend_lookback, momentum_floor, vix_threshold, kelly_fraction,
        TRAIN_START, TRAIN_END
    )

    if result["n_trades"] < MIN_TRADES:
        return -999.0

    return result["cagr"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Running Optuna ({N_TRIALS} trials) — objective: in-market CAGR")
    print(f"Train: {TRAIN_START} to {TRAIN_END}")
    print(f"OOS  : {OOS_START} to {OOS_END}\n")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=50),
        pruner=optuna.pruners.NopPruner(),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False, n_jobs=1)

    best = study.best_trial
    best_params = best.params
    train_cagr  = best.value

    print(f"\nBest train CAGR: {train_cagr*100:.2f}%")
    print(f"Best params: {best_params}")

    # OOS evaluation with best params
    oos_result = evaluate(
        trend_lookback=best_params["trend_lookback"],
        momentum_floor=best_params["momentum_floor"],
        vix_threshold=best_params["vix_threshold"],
        kelly_fraction=best_params["kelly_fraction"],
        date_start=OOS_START,
        date_end=OOS_END,
    )
    oos_cagr = oos_result["cagr"]

    print(f"\nOOS ({OOS_START}–{OOS_END}):")
    print(f"  Trades  : {oos_result['n_trades']}")
    print(f"  Mean ret: {oos_result['mean_ret']*100:.4f}%/week")
    print(f"  CAGR    : {oos_cagr*100:.2f}%")

    # Train details
    train_result = evaluate(
        trend_lookback=best_params["trend_lookback"],
        momentum_floor=best_params["momentum_floor"],
        vix_threshold=best_params["vix_threshold"],
        kelly_fraction=best_params["kelly_fraction"],
        date_start=TRAIN_START,
        date_end=TRAIN_END,
    )

    # Top 10 trials summary
    top10 = sorted(study.trials, key=lambda t: t.value if t.value else -999, reverse=True)[:10]
    top10_list = [
        {"params": t.params, "train_cagr_pct": round(t.value * 100, 2)}
        for t in top10 if t.value and t.value > -900
    ]

    output = {
        "objective": "in-market annualized CAGR (kelly-sized)",
        "basket": BASKET,
        "n_trials": N_TRIALS,
        "min_trades_train": MIN_TRADES,
        "train_window": [str(TRAIN_START), str(TRAIN_END)],
        "oos_window": [str(OOS_START), str(OOS_END)],
        "best_params": best_params,
        "train_cagr_pct": round(train_cagr * 100, 2),
        "train_n_trades": train_result["n_trades"],
        "train_mean_weekly_ret_pct": round(train_result["mean_ret"] * 100, 4),
        "oos_cagr_pct": round(oos_cagr * 100, 2),
        "oos_n_trades": oos_result["n_trades"],
        "oos_mean_weekly_ret_pct": round(oos_result["mean_ret"] * 100, 4),
        "top10_trials": top10_list,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {OUT_PATH}")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"\nTrain CAGR ({TRAIN_START}–{TRAIN_END}): {train_cagr*100:.2f}%")
    print(f"OOS   CAGR ({OOS_START}–{OOS_END}): {oos_cagr*100:.2f}%")
    print(f"Train trades: {train_result['n_trades']}")
    print(f"OOS   trades: {oos_result['n_trades']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
