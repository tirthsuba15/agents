#!/usr/bin/env python3
"""
researcher/mutator.py

Loads current strategy parameters from config.py, generates 5 mutations
(±20% on each key threshold), runs a fast 6-month pandas backtest for each,
ranks by Sharpe, stores top 2 in HydraDB as status='candidate'.

Prints an alert if any mutation beats current config Sharpe by >0.1.

Usage:
    python researcher/mutator.py
"""

import datetime
import sys
from pathlib import Path

# Allow direct execution: python researcher/mutator.py
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf

from researcher.hydra_client import insert_many

# ---------------------------------------------------------------------------
# Current strategy parameters (sourced from config intent — do not import
# config directly to avoid side-effects; values must match config defaults)
# ---------------------------------------------------------------------------
BASE_PARAMS = {
    "OPEX_RETURN_THRESHOLD":       0.029,
    "INTRADAY_MOMENTUM_THRESHOLD": 0.0015,
    "CONVICTION_THRESHOLD":        0.35,
}

MUTATION_FACTOR = 0.20   # ±20%
N_MUTATIONS     = 10
LOOKBACK_MONTHS = 6
BASKET          = ["NVDA", "JNJ", "UNH", "WMT", "HD", "ABBV", "AAPL", "V"]
ALERT_DELTA     = 0.10   # alert if Sharpe improves by this much


# ---------------------------------------------------------------------------
# Fast pandas backtest
# ---------------------------------------------------------------------------

def _fetch_daily(symbols: list[str], months: int = 6) -> pd.DataFrame:
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=months * 31)
    raw   = yf.download(symbols, start=str(start), end=str(end),
                         auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        closes = raw[["Close"]]
    return closes.dropna(how="all")


def _is_opex_week(date: datetime.date) -> bool:
    import calendar
    c       = calendar.monthcalendar(date.year, date.month)
    fridays = [week[calendar.FRIDAY] for week in c if week[calendar.FRIDAY] != 0]
    tf      = datetime.date(date.year, date.month, fridays[2])
    monday  = tf - datetime.timedelta(days=tf.weekday())
    return monday <= date <= tf


def _run_backtest(params: dict, closes: pd.DataFrame) -> float:
    """
    OPEX-week long-only strategy backtest.
    Entry: hold basket equally weighted during OPEX weeks when the prior
           non-OPEX week's basket return > -OPEX_RETURN_THRESHOLD (trend gate)
           AND CONVICTION_THRESHOLD determines minimum position fraction held.
    Exit: all other weeks hold cash.
    Returns annualised Sharpe of weekly strategy returns.
    """
    opex_thresh   = params["OPEX_RETURN_THRESHOLD"]
    conviction    = params["CONVICTION_THRESHOLD"]
    position_frac = conviction / 0.35          # normalised to base conviction

    weekly   = closes.resample("W-FRI").last().pct_change().dropna()
    basket_w = weekly.mean(axis=1)
    dates    = [d.date() if hasattr(d, "date") else d for d in basket_w.index]

    strategy_rets = []
    prior_ret     = 0.0

    for i, (date, ret) in enumerate(zip(dates, basket_w.values)):
        if _is_opex_week(date) and prior_ret > -opex_thresh:
            strategy_rets.append(float(ret) * position_frac)
        else:
            strategy_rets.append(0.0)
        prior_ret = float(ret)

    s = pd.Series(strategy_rets)
    if s.std() == 0:
        return 0.0
    return float((s.mean() / s.std()) * np.sqrt(52))


# ---------------------------------------------------------------------------
# Mutation generation
# ---------------------------------------------------------------------------

def _generate_mutations() -> list[dict]:
    mutations = []
    params_list = list(BASE_PARAMS.items())

    # Single-param mutations (5)
    for i in range(5):
        key, base_val = params_list[i % len(params_list)]
        sign   = 1 if i % 2 == 0 else -1
        factor = 1 + sign * MUTATION_FACTOR
        new_params = dict(BASE_PARAMS)
        new_params[key] = round(base_val * factor, 6)
        mutations.append({
            "params": new_params,
            "label":  f"single_{i+1}: {key}={new_params[key]:.5f}",
            "varied": key,
        })

    # Two-param combination mutations (5)
    import itertools
    param_combos = list(itertools.combinations(params_list, 2))
    for i, ((k1, v1), (k2, v2)) in enumerate(param_combos[:5]):
        sign1 = 1 if i % 2 == 0 else -1
        sign2 = -1 if i % 2 == 0 else 1
        new_params = dict(BASE_PARAMS)
        new_params[k1] = round(v1 * (1 + sign1 * MUTATION_FACTOR), 6)
        new_params[k2] = round(v2 * (1 + sign2 * MUTATION_FACTOR), 6)
        label = f"combo_{i+1}: {k1}={new_params[k1]:.4f}, {k2}={new_params[k2]:.4f}"
        mutations.append({
            "params": new_params,
            "label":  label,
            "varied": f"{k1}+{k2}",
        })

    return mutations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_mutations() -> list[dict]:
    print(f"  Fetching {LOOKBACK_MONTHS}-month OHLCV for {BASKET}...")
    closes = _fetch_daily(BASKET, months=LOOKBACK_MONTHS)
    if closes.empty:
        print("  [mutator] No data fetched.", file=sys.stderr)
        return []

    # Baseline Sharpe
    baseline_sharpe = _run_backtest(BASE_PARAMS, closes)
    print(f"  Baseline Sharpe : {baseline_sharpe:.4f}")
    print(f"  Base params     : {BASE_PARAMS}")
    print()

    mutations = _generate_mutations()
    results   = []

    for m in mutations:
        sharpe = _run_backtest(m["params"], closes)
        delta  = sharpe - baseline_sharpe
        tag    = "IMPROVED" if delta > 0 else "WORSE"
        print(f"  {m['label']:<55}  Sharpe={sharpe:.4f}  delta={delta:+.4f}  [{tag}]")
        results.append({
            "label":     m["label"],
            "params":    m["params"],
            "sharpe":    round(sharpe, 4),
            "delta":     round(delta, 4),
            "varied":    m["varied"],
        })

    results.sort(key=lambda x: -x["sharpe"])

    improved = [r for r in results if r["delta"] > 0]
    print(f"\n  {len(improved)}/{len(results)} mutations beat baseline")
    if improved:
        print(f"  Best improvement: +{improved[0]['delta']:.4f} Sharpe via {improved[0]['varied']}")

    top2 = results[:5]

    # Alert if top mutation beats baseline by ALERT_DELTA
    if top2 and top2[0]["delta"] > ALERT_DELTA:
        print(f"\n  [ALERT] Mutation '{top2[0]['label']}' beats baseline by "
              f"{top2[0]['delta']:+.4f} Sharpe — consider promoting to config")

    # Store top 2 in HydraDB
    records = []
    for r in top2:
        records.append({
            "name":               r["label"],
            "signal_description": f"Parameter mutation: {r['varied']} adjusted ±20%",
            "entry_rule":         f"OPEX week with basket return > {r['params']['OPEX_RETURN_THRESHOLD']}",
            "exit_rule":          "End of OPEX week",
            "estimated_sharpe":   r["sharpe"],
            "data_requirements":  ["daily OHLCV", "OPEX calendar"],
            "source":             "mutator",
            "status":             "candidate",
            "params_json":        str(r["params"]),
        })

    print(f"\n  Storing top {len(records)} mutations in HydraDB...")
    stored = insert_many(records)
    print(f"  {stored}/{len(records)} stored")
    return top2


def main() -> None:
    print("Mutator — 5 parameter mutations, 6-month backtest")
    print(f"  Basket: {BASKET}")
    top = run_mutations()
    if top:
        print(f"\n  Top mutation: {top[0]['label']}  Sharpe={top[0]['sharpe']:.4f}")
    else:
        print("  No mutations completed.")


if __name__ == "__main__":
    main()
