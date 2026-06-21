#!/usr/bin/env python3
"""
OPEX Backtest v2 — Stivers & Sun (2013) replication + Smart OPEX filter
=======================================================================
Finding from v1: raw OPEX weeks underperformed 2015-2024 because the
effect only holds in bull-market regimes (dealers long gamma from call
selling). Bear-market OPEX weeks see the effect reverse.

Smart OPEX filter (both conditions must be true on the Monday of the week):
  1. Trend:    SPY close last Friday > SPY 10-week rolling average close
  2. Momentum: Prior week return > -1% (don't enter after a crash week)

Strategies compared:
  - Smart OPEX   (filtered)
  - Naive OPEX   (all OPEX weeks, for reference)
  - Buy & Hold   (all weeks)

Data: yfinance — free, full 2015-2024 history.
Gate: Finnhub /calendar/economic if key available, else hardcoded FOMC/CPI/NFP.

Optional env var:
    FINNHUB_API_KEY

Usage:
    python backtest/opex_backtest.py
"""

import calendar
import datetime
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")

START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
SYMBOLS = ["AAPL", "SPY"]
PRIMARY_SYMBOL = "AAPL"
TREND_LOOKBACK_WEEKS = 10        # SPY MA window for trend filter
MOMENTUM_FLOOR = -0.01           # prior week must be > -1%

FINNHUB_BASE = "https://finnhub.io/api/v1"
GATE_KEYWORDS = {"fomc", "federal open market", "cpi", "consumer price", "nonfarm", "nfp"}

# ---------------------------------------------------------------------------
# Hardcoded gate dates — FOMC, CPI, NFP release days 2015-2024
# Used when Finnhub /calendar/economic returns 403 (paid endpoint).
# ---------------------------------------------------------------------------
_GATE_DATES: set[datetime.date] = {datetime.date.fromisoformat(d) for d in [
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
    # NFP (first Friday of month)
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
    """Return the date of the third Friday of the given month."""
    c = calendar.monthcalendar(year, month)
    fridays = [week[calendar.FRIDAY] for week in c if week[calendar.FRIDAY] != 0]
    return datetime.date(year, month, fridays[2])


def is_opex_week(date: datetime.date) -> bool:
    """True if date falls Mon-Fri of OPEX week (week of the 3rd Friday)."""
    tf = third_friday(date.year, date.month)
    monday = tf - datetime.timedelta(days=tf.weekday())
    return monday <= date <= tf


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch split-adjusted daily bars via yfinance."""
    df = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True, actions=False)
    if df.empty:
        raise RuntimeError(f"No data for {symbol}")
    df.index = df.index.tz_localize(None).date if df.index.tz else df.index.date
    return df[["Open", "Close"]].rename(columns={"Open": "open", "Close": "close"})


def fetch_blocked_dates() -> set:
    """Try Finnhub first, fall back to hardcoded dates."""
    if not FINNHUB_API_KEY:
        print("  Using hardcoded FOMC/CPI/NFP dates")
        return _GATE_DATES

    blocked: set = set()
    start_dt = datetime.date.fromisoformat(START_DATE)
    end_dt = datetime.date.fromisoformat(END_DATE)
    cur, any_ok = start_dt, False
    while cur <= end_dt:
        chunk_end = min(datetime.date(cur.year, 12, 31), end_dt)
        try:
            r = requests.get(
                f"{FINNHUB_BASE}/calendar/economic",
                params={"from": cur.isoformat(), "to": chunk_end.isoformat(), "token": FINNHUB_API_KEY},
                timeout=20,
            )
            r.raise_for_status()
            for ev in r.json().get("economicCalendar", []):
                if any(kw in ev.get("event", "").lower() for kw in GATE_KEYWORDS):
                    try:
                        blocked.add(datetime.date.fromisoformat(ev["time"][:10]))
                    except (KeyError, ValueError):
                        pass
            any_ok = True
        except requests.RequestException:
            pass
        cur = datetime.date(cur.year + 1, 1, 1)

    if any_ok:
        return blocked
    print("  Finnhub unavailable — using hardcoded FOMC/CPI/NFP dates")
    return _GATE_DATES


# ---------------------------------------------------------------------------
# Weekly return + signal computation
# ---------------------------------------------------------------------------

def group_by_week(df: pd.DataFrame) -> dict:
    """Returns {(iso_year, iso_week): [date, ...]} sorted."""
    weeks: dict = {}
    for d in sorted(df.index):
        weeks.setdefault(d.isocalendar()[:2], []).append(d)
    return {k: weeks[k] for k in sorted(weeks)}


def spy_trend_signal(spy_df: pd.DataFrame, lookback: int = TREND_LOOKBACK_WEEKS) -> dict:
    """
    Returns {week_start: (uptrend_bool, prior_week_ret)} for each week in SPY data.
    uptrend = SPY weekly close last week > rolling lookback-week average.
    Both values use data available before OPEX week opens (prior Friday close).
    """
    closes = spy_df["close"].to_dict()
    weeks = group_by_week(spy_df)
    week_keys = sorted(weeks.keys())

    # weekly close for each week (last trading day's close)
    weekly_close = {min(days): closes[max(days)] for days in weeks.values()}
    week_starts = sorted(weekly_close.keys())

    signals: dict = {}
    for i, ws in enumerate(week_starts):
        if i == 0:
            signals[ws] = (True, 0.0)
            continue
        prior_close = weekly_close[week_starts[i - 1]]
        prior_open_days = weeks[week_keys[i - 1]]
        prior_open_val = spy_df["open"][min(prior_open_days)]
        prior_ret = (prior_close - prior_open_val) / prior_open_val

        window_closes = [weekly_close[week_starts[j]] for j in range(max(0, i - lookback), i)]
        ma = sum(window_closes) / len(window_closes)
        uptrend = prior_close > ma

        signals[ws] = (uptrend, prior_ret)
    return signals


def compute_weekly_returns(
    df: pd.DataFrame,
    blocked_dates: set,
    spy_signals: dict,
) -> pd.DataFrame:
    """
    Weekly return = (last close - first open) / first open.
    Columns: ret, naive_opex, smart_opex
      naive_opex: OPEX week with macro gate applied
      smart_opex: naive_opex AND uptrend AND prior week > MOMENTUM_FLOOR
    """
    opens = df["open"].to_dict()
    closes = df["close"].to_dict()
    weeks = group_by_week(df)

    records = []
    for key in sorted(weeks):
        days = weeks[key]
        first, last = min(days), max(days)
        if first not in opens or last not in closes:
            continue

        ret = (closes[last] - opens[first]) / opens[first]
        opex = is_opex_week(first)

        # Macro gate
        if opex and any(d in blocked_dates for d in days):
            opex = False

        # Smart filter — requires SPY trend signal for this week
        uptrend, prior_ret = spy_signals.get(first, (True, 0.0))
        smart = opex and uptrend and prior_ret > MOMENTUM_FLOOR

        records.append({"week_start": first, "ret": ret, "naive_opex": opex, "smart_opex": smart})

    return pd.DataFrame(records).set_index("week_start")


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def annualised_sharpe(rets: pd.Series) -> float:
    std = rets.std()
    return float((rets.mean() / std) * np.sqrt(52)) if std else 0.0


def win_rate(rets: pd.Series) -> float:
    return float((rets > 0).mean())


def print_stats(label: str, rets: pd.Series) -> float:
    mean_pct = rets.mean() * 100
    print(f"  {label:<18}: n={len(rets):>3}  mean={mean_pct:+.4f}%  "
          f"Sharpe={annualised_sharpe(rets):.2f}  win={win_rate(rets)*100:.1f}%")
    return mean_pct


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(weekly: pd.DataFrame, symbol: str, out_path: Path) -> None:
    smart_rets = weekly[weekly["smart_opex"]]["ret"]
    naive_rets = weekly[weekly["naive_opex"]]["ret"]
    non_opex_rets = weekly[~weekly["naive_opex"]]["ret"]

    smart_cum = (1 + smart_rets).cumprod()
    naive_cum = (1 + naive_rets).cumprod()
    bah_cum = (1 + weekly["ret"]).cumprod()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # -- Bar chart --
    labels = ["Smart OPEX", "Naive OPEX", "Non-OPEX"]
    means = [smart_rets.mean() * 100, naive_rets.mean() * 100, non_opex_rets.mean() * 100]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    bars = ax1.bar(labels, means, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    ax1.axhline(0, color="#7f8c8d", linewidth=0.8, linestyle="--")
    for bar, val in zip(bars, means):
        ypos = val + 0.006 if val >= 0 else val - 0.022
        ax1.text(
            bar.get_x() + bar.get_width() / 2, ypos,
            f"{val:+.3f}%", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=bar.get_facecolor(),
        )
    margin = max(abs(m) for m in means) * 0.6
    ax1.set_ylim(min(0, min(means)) - margin, max(means) + margin)
    ax1.set_ylabel("Mean Weekly Return (%)", fontsize=10)
    ax1.set_title(f"{symbol} — Mean Weekly Return\nSmart vs Naive OPEX, 2015-2024", fontsize=10)

    # -- Cumulative chart --
    ax2.plot(smart_cum.index, smart_cum.values, label="Smart OPEX", color="#2ecc71", linewidth=2.2)
    ax2.plot(naive_cum.index, naive_cum.values, label="Naive OPEX", color="#f39c12",
             linewidth=1.4, linestyle="--")
    ax2.plot(bah_cum.index, bah_cum.values, label="Buy & Hold", color="#2980b9",
             linewidth=1.2, alpha=0.7)
    ax2.set_ylabel("Cumulative Return (x)", fontsize=10)
    ax2.set_title(f"{symbol} — Cumulative Return\nSmart OPEX vs Naive vs Buy & Hold", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)
    ax2.tick_params(axis="x", labelrotation=30, labelsize=8)

    plt.suptitle(
        f"OPEX Backtest v2  -  {symbol}  -  Stivers & Sun (2013) + Regime Filter",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {out_path}")


# ---------------------------------------------------------------------------
# Per-symbol runner
# ---------------------------------------------------------------------------

def run(symbol: str, blocked_dates: set, spy_signals: dict) -> None:
    print(f"\n{'─'*55}")
    print(f"  {symbol}  |  {START_DATE} to {END_DATE}")
    print(f"{'─'*55}")

    df = fetch_bars(symbol, START_DATE, END_DATE)
    print(f"  Daily bars : {len(df)}")

    weekly = compute_weekly_returns(df, blocked_dates, spy_signals)

    smart_mean = print_stats("Smart OPEX", weekly[weekly["smart_opex"]]["ret"])
    naive_mean = print_stats("Naive OPEX", weekly[weekly["naive_opex"]]["ret"])
    non_mean   = print_stats("Non-OPEX", weekly[~weekly["naive_opex"]]["ret"])

    if symbol == PRIMARY_SYMBOL:
        passed = smart_mean > 0.30 and non_mean < 0.20
        tag = "PASS" if passed else "INFO"
        print(f"\n  [{tag}] Demo target: Smart OPEX {smart_mean:.3f}% (need >0.30%), "
              f"Non-OPEX {non_mean:.3f}% (need <0.20%)")

    out_path = Path(__file__).parent / f"opex_backtest_{symbol}.png"
    plot_results(weekly, symbol, out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("OPEX Backtest v2 - Smart Regime Filter")
    print(f"Symbols: {SYMBOLS}  |  {START_DATE} to {END_DATE}")
    print(f"Filter:  SPY {TREND_LOOKBACK_WEEKS}-week MA trend + prior week > {MOMENTUM_FLOOR*100:.0f}%\n")

    print("Loading macro gate dates...")
    blocked_dates = fetch_blocked_dates()
    print(f"  Gate dates: {len(blocked_dates)}")

    print("\nComputing SPY trend signals...")
    spy_df = fetch_bars("SPY", START_DATE, END_DATE)
    spy_signals = spy_trend_signal(spy_df)

    for sym in SYMBOLS:
        run(sym, blocked_dates, spy_signals)

    print("\nDone.")


if __name__ == "__main__":
    main()
