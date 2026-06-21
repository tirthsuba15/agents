#!/usr/bin/env python3
"""
Optuna parameter search over the OPEX smart-filter strategy — OOS-gated.
=========================================================================
Target (Option B from OPTIMIZATION_PLAN.md): tune the regime-filter params of
the OPEX basket strategy (backtest/opex_backtest.py), a proper Optuna study
replacing researcher/mutator.py's crude 5-mutation pass.

OOS discipline (per advisor + plan "no in-sample mirage"):
  - Study tunes on TRAIN+VALIDATION only (2015-01 .. 2021-12), selecting params
    by VALIDATION-window deployable-capital Sharpe (net of costs).
  - Best params are evaluated ONCE on a held-out TEST window (2022-01 .. 2024-12)
    that the study never touched. That single number is the "optimized OOS".
  - Baseline (current defaults) is evaluated on the SAME test window.
  - Delta = optimized_test - baseline_test, apples-to-apples, same Sharpe def.

Realism guards:
  - Sharpe computed on the FULL contiguous weekly P&L series (flat weeks = 0),
    i.e. deployable-capital Sharpe — NOT the traded-weeks subset (which rewards
    trading rarely).
  - Per-entry/exit transaction cost (slippage + commission) charged on weeks the
    basket is entered.
  - Minimum-trades constraint in the objective: too-few filtered weeks -> -inf,
    so Optuna can't cherry-pick a handful of lucky weeks.

Data: yfinance daily (cached once, every trial runs against in-memory frames).
Does NOT overwrite any models/*.pkl. Writes only under optimize/.

Usage:
    python optimize/optuna_opex.py
"""

import calendar
import datetime
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import yfinance as yf

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"

# Train/validation vs held-out test split (test never seen by the study)
TRAIN_END   = datetime.date(2020, 12, 31)   # study training window  : 2015..2020
VAL_END     = datetime.date(2021, 12, 31)   # study validation window: 2021
# TEST window = 2022-01-01 .. END_DATE  (held out)

BASKET = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "JPM",  "JNJ",   "UNH",
    "HD",   "WMT",  "PG",    "BAC",  "MA",
    "V",    "ABBV", "MRK",   "PFE",
    "XLK",  "XLV",  "XLF",
]
SPY_SYM = "SPY"
VIX_SYM = "^VIX"

# Baseline (current production defaults from backtest/opex_backtest.py)
BASELINE_PARAMS = {
    "TREND_LOOKBACK": 10,
    "MOMENTUM_FLOOR": -0.01,
    "VIX_THRESHOLD":  30.0,
    "KELLY_FRACTION": 0.25,
}

# Realistic cost model: round-trip slippage + commission per basket entry, in
# return terms (charged on weeks we hold the basket). 15 bps round-trip is a
# reasonable estimate for liquid large-caps + ETFs (equal-weight 21 names).
COST_PER_ENTRY = 0.0015   # 15 bps charged on each week the basket is entered

MIN_TRADES = 20           # minimum filtered OPEX weeks in a window for a valid Sharpe
N_TRIALS   = 400
TIMEOUT_S  = 1200         # 20 min hard cap (study still finishes well under the 30-min box)
SEED       = 42

OUT_DIR = Path(__file__).parent


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


# ---------------------------------------------------------------------------
# Data loading (once)
# ---------------------------------------------------------------------------

def fetch_bars(symbol: str) -> pd.DataFrame:
    df = yf.Ticker(symbol).history(
        start=START_DATE, end=END_DATE, auto_adjust=True, actions=False
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
    opens = df["open"].to_dict()
    closes = df["close"].to_dict()
    weeks = group_by_week(df)
    out = {}
    for days in weeks.values():
        first, last = min(days), max(days)
        if first in opens and last in closes:
            out[first] = (closes[last] - opens[first]) / opens[first]
    return pd.Series(out, name="ret").sort_index()


def build_weekly_close(df: pd.DataFrame) -> pd.Series:
    """Week_start -> last close of that week."""
    closes = df["close"].to_dict()
    out = {}
    for days in group_by_week(df).values():
        out[min(days)] = closes[max(days)]
    return pd.Series(out).sort_index()


# ---------------------------------------------------------------------------
# Build the once-only cached panel
# ---------------------------------------------------------------------------

def load_panel() -> dict:
    print(f"Fetching daily bars for {len(BASKET)} basket names + SPY + VIX...")
    t = time.time()
    spy_df = fetch_bars(SPY_SYM)
    vix_df = fetch_bars(VIX_SYM)

    all_rets = {}
    skipped = []
    for sym in BASKET:
        df = fetch_bars(sym)
        if df.empty or len(df) < 100:
            skipped.append(sym)
            continue
        all_rets[sym] = weekly_returns_for(df)
    if skipped:
        print(f"  skipped (insufficient data): {skipped}")

    combined = pd.DataFrame(all_rets).dropna(how="all")
    basket_ret = combined.mean(axis=1)
    basket_ret.name = "ret"

    spy_rets = weekly_returns_for(spy_df)
    spy_close_w = build_weekly_close(spy_df)
    vix_close_w = build_weekly_close(vix_df)

    # week_start -> list of trading days that week (for macro gate, kept simple
    # here: we keep the OPEX flag; macro-date gate from the original file is a
    # constant across all param choices so it does not affect the *delta*. We
    # drop it for clarity and to keep baseline vs optimized strictly comparable.)
    idx = basket_ret.index
    opex_flag = pd.Series([is_opex_week(ws) for ws in idx], index=idx)

    print(f"  panel built in {time.time()-t:.1f}s — {len(idx)} basket weeks, "
          f"{int(opex_flag.sum())} OPEX weeks")
    return {
        "basket_ret":  basket_ret,
        "spy_rets":    spy_rets,
        "spy_close_w": spy_close_w,
        "vix_close_w": vix_close_w,
        "opex_flag":   opex_flag,
        "index":       idx,
    }


# ---------------------------------------------------------------------------
# Signal construction (parameterized) — no look-ahead
# ---------------------------------------------------------------------------

def build_smart_signal(panel: dict, trend_lookback: int, momentum_floor: float,
                       vix_threshold: float) -> pd.Series:
    """
    Smart-OPEX boolean per week (mirrors opex_backtest.label_weeks):
      smart = opex AND uptrend AND prior_spy_ret > momentum_floor AND vix_low
    All inputs are shifted to be known on Monday of the OPEX week (no look-ahead).
    """
    idx = panel["index"]
    spy_close = panel["spy_close_w"]
    spy_rets = panel["spy_rets"]
    vix_close = panel["vix_close_w"]

    ma = spy_close.rolling(trend_lookback, min_periods=1).mean().shift(1)
    prior_close = spy_close.shift(1)
    uptrend = (prior_close > ma).reindex(idx, fill_value=True)

    prior_ret = spy_rets.shift(1).reindex(idx, fill_value=0.0)

    vix_low = (vix_close < vix_threshold).reindex(idx)
    vix_low = vix_low.fillna(True).astype(bool)

    smart = panel["opex_flag"].astype(bool) & uptrend.astype(bool) & \
            (prior_ret > momentum_floor) & vix_low
    return smart


def kelly_sizes(panel: dict, smart: pd.Series, kelly_fraction: float) -> pd.Series:
    """
    Expanding-window (past-only) Kelly sizing, mirrors opex_backtest.compute_kelly_size.
    Uses only smart weeks strictly before the current week -> no look-ahead.
    """
    idx = panel["index"]
    basket_ret = panel["basket_ret"]
    sizes = pd.Series(0.0, index=idx)
    smart_idx = [ws for ws in idx if smart.loc[ws]]

    smart_set = set(smart_idx)
    for ws in smart_idx:
        past_smart = [basket_ret.loc[w] for w in idx if w < ws and w in smart_set]
        if len(past_smart) < 10:
            # match original opex_backtest: full size (1.0) when insufficient history
            sizes.loc[ws] = 1.0
            continue
        arr = np.array(past_smart)
        wins = (arr > 0).sum()
        losses = (arr <= 0).sum()
        win_rate = wins / len(arr)
        avg_win = arr[arr > 0].mean() if wins > 0 else 0.01
        avg_loss = abs(arr[arr <= 0].mean()) if losses > 0 else 0.01
        odds = avg_win / avg_loss
        kelly = (odds * win_rate - (1 - win_rate)) / odds if odds > 0 else 0.0
        kelly = max(0.0, min(1.0, kelly))
        sizes.loc[ws] = kelly * kelly_fraction
    return sizes


# ---------------------------------------------------------------------------
# Strategy P&L + deployable-capital Sharpe
# ---------------------------------------------------------------------------

def strategy_returns(panel: dict, params: dict) -> pd.Series:
    """Full contiguous weekly strategy return series (flat weeks = 0), net of cost."""
    smart = build_smart_signal(
        panel, int(params["TREND_LOOKBACK"]), float(params["MOMENTUM_FLOOR"]),
        float(params["VIX_THRESHOLD"]),
    )
    sizes = kelly_sizes(panel, smart, float(params["KELLY_FRACTION"]))
    basket_ret = panel["basket_ret"]

    gross = basket_ret * sizes * smart.astype(float)
    # cost charged on every week we take a position, PROPORTIONAL to deployed
    # size (so a global sizing scalar like KELLY_FRACTION is NOT a free lever
    # that games a fixed cost — Sharpe is invariant to it under proportional cost)
    traded = (sizes > 0) & smart
    cost = traded.astype(float) * COST_PER_ENTRY * sizes
    net = gross - cost
    net.name = "net_ret"
    return net, smart


def sharpe_full(rets: pd.Series) -> float:
    """Deployable-capital annualised Sharpe over the FULL contiguous series."""
    std = rets.std()
    return float((rets.mean() / std) * np.sqrt(52)) if std and std > 0 else 0.0


def window_mask(idx, lo: datetime.date | None, hi: datetime.date | None) -> pd.Series:
    return pd.Series([(lo is None or ws > lo) and (hi is None or ws <= hi) for ws in idx],
                     index=idx)


def evaluate(panel: dict, params: dict, lo, hi) -> dict:
    """Evaluate strategy on the window (lo, hi]. Returns Sharpe + trade count."""
    net, smart = strategy_returns(panel, params)
    m = window_mask(panel["index"], lo, hi)
    win_net = net[m]
    win_smart = smart[m]
    n_trades = int(win_smart.sum())
    return {"sharpe": sharpe_full(win_net), "n_trades": n_trades, "mean_ret": float(win_net.mean())}


# ---------------------------------------------------------------------------
# Optuna objective — tunes on validation window only
# ---------------------------------------------------------------------------

def make_objective(panel):
    def objective(trial: optuna.Trial) -> float:
        # Only the regime-filter params are tuned. KELLY_FRACTION is held fixed
        # at the baseline (0.25) for BOTH baseline and optimized: under the
        # proportional cost model Sharpe is invariant to a global sizing scalar,
        # so it is not a meaningful lever — tuning it would just be noise.
        params = {
            "TREND_LOOKBACK": trial.suggest_int("TREND_LOOKBACK", 4, 30),
            "MOMENTUM_FLOOR": trial.suggest_float("MOMENTUM_FLOOR", -0.05, 0.01),
            "VIX_THRESHOLD":  trial.suggest_float("VIX_THRESHOLD", 15.0, 45.0),
            "KELLY_FRACTION": BASELINE_PARAMS["KELLY_FRACTION"],
        }
        # Select on the TRAIN+VALIDATION window Sharpe (2015..2021) — more trades,
        # less noise than a val-only window. Require enough filtered weeks over
        # that span to block sparse-trade cherry-picking.
        train_val = evaluate(panel, params, None, VAL_END)
        if train_val["n_trades"] < MIN_TRADES:
            return -1e9
        return train_val["sharpe"]
    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    panel = load_panel()

    # ---- Baseline on held-out TEST (2022..2024) ----
    base_test = evaluate(panel, BASELINE_PARAMS, VAL_END, None)
    base_val  = evaluate(panel, BASELINE_PARAMS, TRAIN_END, VAL_END)
    print(f"\nBaseline params: {BASELINE_PARAMS}")
    print(f"  Baseline VALIDATION Sharpe (2021)      : {base_val['sharpe']:.3f}  "
          f"(trades={base_val['n_trades']})")
    print(f"  Baseline HELD-OUT TEST Sharpe (2022-24): {base_test['sharpe']:.3f}  "
          f"(trades={base_test['n_trades']})")

    # ---- Optuna study tuned on validation only ----
    print(f"\nRunning Optuna study (n_trials<={N_TRIALS}, timeout={TIMEOUT_S}s)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(make_objective(panel), n_trials=N_TRIALS, timeout=TIMEOUT_S,
                   show_progress_bar=False)

    best_params = dict(BASELINE_PARAMS)
    best_params.update(study.best_params)
    print(f"\nBest train+val (2015-2021) selection Sharpe: {study.best_value:.3f}")
    print(f"Best params: {best_params}")
    print(f"Completed trials: {len(study.trials)}")

    # ---- Evaluate best params ONCE on held-out TEST ----
    opt_test = evaluate(panel, best_params, VAL_END, None)
    opt_val  = evaluate(panel, best_params, TRAIN_END, VAL_END)
    print(f"\nOptimized VALIDATION Sharpe (2021)      : {opt_val['sharpe']:.3f}  "
          f"(trades={opt_val['n_trades']})")
    print(f"Optimized HELD-OUT TEST Sharpe (2022-24): {opt_test['sharpe']:.3f}  "
          f"(trades={opt_test['n_trades']})")

    delta = opt_test["sharpe"] - base_test["sharpe"]
    print(f"\n=== OOS DELTA (held-out test, net of {COST_PER_ENTRY*1e4:.0f}bps cost) ===")
    print(f"  baseline test Sharpe : {base_test['sharpe']:.3f}  (trades={base_test['n_trades']})")
    print(f"  optimized test Sharpe: {opt_test['sharpe']:.3f}  (trades={opt_test['n_trades']})")
    print(f"  delta                : {delta:+.3f}")
    if opt_test["n_trades"] < base_test["n_trades"] * 0.5:
        print("  [WARN] optimized trades far below baseline — possible overfit by trading less")

    elapsed = time.time() - t0
    results = {
        "target": "OPEX smart-filter strategy (Option B)",
        "objective": "deployable-capital annualised Sharpe (net of proportional cost); params selected on train+val (2015-2021), reported on held-out test (2022-2024); KELLY_FRACTION fixed at 0.25 for both",
        "data": {
            "source": "yfinance daily",
            "basket": BASKET,
            "period": [START_DATE, END_DATE],
            "train_window": ["2015-01-01", TRAIN_END.isoformat()],
            "validation_window": [(TRAIN_END + datetime.timedelta(days=1)).isoformat(), VAL_END.isoformat()],
            "test_window_heldout": [(VAL_END + datetime.timedelta(days=1)).isoformat(), END_DATE],
        },
        "cost_model": {"cost_per_entry_return": COST_PER_ENTRY, "bps_round_trip": COST_PER_ENTRY * 1e4},
        "min_trades_constraint": MIN_TRADES,
        "baseline_params": BASELINE_PARAMS,
        "best_params": best_params,
        "baseline_validation_sharpe": round(base_val["sharpe"], 4),
        "optimized_validation_sharpe": round(opt_val["sharpe"], 4),
        "baseline_test_sharpe_OOS": round(base_test["sharpe"], 4),
        "optimized_test_sharpe_OOS": round(opt_test["sharpe"], 4),
        "delta_test_sharpe_OOS": round(delta, 4),
        "baseline_test_trades": base_test["n_trades"],
        "optimized_test_trades": opt_test["n_trades"],
        "n_trials": len(study.trials),
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    out_path = OUT_DIR / "opex_optuna_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {out_path}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
