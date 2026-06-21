#!/usr/bin/env python3
"""
OPEX Backtest — Stivers & Sun (2013) replication
OPEX weeks (Mon open -> 3rd Fri close) vs non-OPEX weeks for AAPL & SPY 2015-2024.
Gate: skip OPEX weeks containing FOMC, CPI, or NFP events.

Data source: yfinance (free, full 2015-2024 history).
Economic gate: Finnhub /calendar/economic if key provided, else hardcoded
               FOMC/CPI/NFP dates scraped from public records (2015-2024).

Optional env vars:
    FINNHUB_API_KEY  — upgrades gate to live Finnhub data

Usage:
    python backtest/opex_backtest.py
"""

import calendar
import datetime
import os
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
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

FINNHUB_BASE = "https://finnhub.io/api/v1"
GATE_KEYWORDS = {"fomc", "federal open market", "cpi", "consumer price", "nonfarm", "nfp"}

# ---------------------------------------------------------------------------
# Hardcoded gate dates (FOMC, CPI, NFP) 2015-2024
# Source: Federal Reserve meeting calendars + BLS release schedules
# Only dates that fall inside an OPEX week matter, but we store all for accuracy.
# ---------------------------------------------------------------------------
_HARDCODED_GATE_DATES: list[str] = [
    # FOMC 2015
    "2015-01-28","2015-03-18","2015-04-29","2015-06-17","2015-07-29",
    "2015-09-17","2015-10-28","2015-12-16",
    # FOMC 2016
    "2016-01-27","2016-03-16","2016-04-27","2016-06-15","2016-07-27",
    "2016-09-21","2016-11-02","2016-12-14",
    # FOMC 2017
    "2017-02-01","2017-03-15","2017-05-03","2017-06-14","2017-07-26",
    "2017-09-20","2017-11-01","2017-12-13",
    # FOMC 2018
    "2018-01-31","2018-03-21","2018-05-02","2018-06-13","2018-08-01",
    "2018-09-26","2018-11-08","2018-12-19",
    # FOMC 2019
    "2019-01-30","2019-03-20","2019-05-01","2019-06-19","2019-07-31",
    "2019-09-18","2019-10-30","2019-12-11",
    # FOMC 2020
    "2020-01-29","2020-03-03","2020-03-15","2020-04-29","2020-06-10",
    "2020-07-29","2020-09-16","2020-11-05","2020-12-16",
    # FOMC 2021
    "2021-01-27","2021-03-17","2021-04-28","2021-06-16","2021-07-28",
    "2021-09-22","2021-11-03","2021-12-15",
    # FOMC 2022
    "2022-01-26","2022-03-16","2022-05-04","2022-06-15","2022-07-27",
    "2022-09-21","2022-11-02","2022-12-14",
    # FOMC 2023
    "2023-02-01","2023-03-22","2023-05-03","2023-06-14","2023-07-26",
    "2023-09-20","2023-11-01","2023-12-13",
    # FOMC 2024
    "2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31",
    "2024-09-18","2024-11-07","2024-12-18",
    # CPI release days (BLS, typically 2nd or 3rd week of month)
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
    # NFP (first Friday of month, BLS)
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
]


# ---------------------------------------------------------------------------
# OPEX helpers
# ---------------------------------------------------------------------------

def third_friday(year: int, month: int) -> datetime.date:
    """Return the date of the third Friday of the given month."""
    c = calendar.monthcalendar(year, month)
    fridays = [week[calendar.FRIDAY] for week in c if week[calendar.FRIDAY] != 0]
    return datetime.date(year, month, fridays[2])


def is_opex_week(date: datetime.date) -> bool:
    """True if date falls Mon-Fri of OPEX week (week containing the 3rd Friday)."""
    tf = third_friday(date.year, date.month)
    monday = tf - datetime.timedelta(days=tf.weekday())  # Friday.weekday()==4
    return monday <= date <= tf


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch split-adjusted daily bars via yfinance (free, full 2015-2024 history)."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, auto_adjust=True, actions=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {symbol}")
    df.index = df.index.tz_localize(None).date if df.index.tz is not None else df.index.date
    return df[["Open", "Close"]].rename(columns={"Open": "open", "Close": "close"})


def fetch_blocked_dates(start: str, end: str) -> set:
    """
    Return set of dates containing FOMC, CPI, or NFP events.
    Tries Finnhub first; falls back to hardcoded dates.
    """
    if FINNHUB_API_KEY:
        blocked: set = set()
        start_dt = datetime.date.fromisoformat(start)
        end_dt = datetime.date.fromisoformat(end)
        cur = start_dt
        success = False
        while cur <= end_dt:
            chunk_end = min(datetime.date(cur.year, 12, 31), end_dt)
            try:
                resp = requests.get(
                    f"{FINNHUB_BASE}/calendar/economic",
                    params={"from": cur.isoformat(), "to": chunk_end.isoformat(), "token": FINNHUB_API_KEY},
                    timeout=20,
                )
                resp.raise_for_status()
                events = resp.json().get("economicCalendar", [])
                for ev in events:
                    name = ev.get("event", "").lower()
                    if any(kw in name for kw in GATE_KEYWORDS):
                        try:
                            blocked.add(datetime.date.fromisoformat(ev["time"][:10]))
                        except (KeyError, ValueError):
                            pass
                success = True
            except requests.RequestException as exc:
                print(f"  Finnhub {cur.year} failed ({exc}), using hardcoded dates")
            cur = datetime.date(cur.year + 1, 1, 1)
        if success:
            return blocked

    print("  Using hardcoded FOMC/CPI/NFP gate dates (2015-2024)")
    return {datetime.date.fromisoformat(d) for d in _HARDCODED_GATE_DATES}


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def compute_weekly_returns(df: pd.DataFrame, blocked_dates: set) -> pd.DataFrame:
    """
    Group trading days by ISO year-week.
    Weekly return = (last close - first open) / first open.
    OPEX weeks that contain a gating event are reclassified as non-OPEX.
    """
    opens = df["open"].to_dict()
    closes = df["close"].to_dict()

    weeks: dict = {}
    for d in sorted(df.index):
        key = d.isocalendar()[:2]
        weeks.setdefault(key, []).append(d)

    records = []
    for key in sorted(weeks):
        days = weeks[key]
        first, last = min(days), max(days)
        if first not in opens or last not in closes:
            continue

        ret = (closes[last] - opens[first]) / opens[first]
        opex = is_opex_week(first)
        if opex and any(d in blocked_dates for d in days):
            opex = False

        records.append({"week_start": first, "opex": opex, "ret": ret})

    return pd.DataFrame(records).set_index("week_start")


def annualised_sharpe(returns: pd.Series, periods_per_year: int = 52) -> float:
    std = returns.std()
    return float((returns.mean() / std) * np.sqrt(periods_per_year)) if std else 0.0


def win_rate(returns: pd.Series) -> float:
    return float((returns > 0).mean())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(weekly: pd.DataFrame, symbol: str, out_path: Path) -> None:
    opex_rets = weekly[weekly["opex"]]["ret"]
    non_opex_rets = weekly[~weekly["opex"]]["ret"]

    opex_cum = (1 + opex_rets).cumprod()
    bah_cum = (1 + weekly["ret"]).cumprod()

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, wspace=0.35)

    # Bar chart
    ax1 = fig.add_subplot(gs[0])
    labels = ["OPEX weeks", "Non-OPEX weeks"]
    means = [opex_rets.mean() * 100, non_opex_rets.mean() * 100]
    colors = ["#27ae60", "#e74c3c"]
    bars = ax1.bar(labels, means, color=colors, width=0.45, edgecolor="white", linewidth=1.2)
    ax1.axhline(0, color="#7f8c8d", linewidth=0.8, linestyle="--")
    for bar, val in zip(bars, means):
        ypos = val + 0.008 if val >= 0 else val - 0.025
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            ypos,
            f"{val:+.3f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=bar.get_facecolor(),
        )
    ax1.set_ylabel("Mean Weekly Return (%)", fontsize=10)
    ax1.set_title(f"{symbol} — Mean Weekly Return\n(OPEX vs Non-OPEX, 2015-2024)", fontsize=10)
    margin = max(abs(m) for m in means) * 0.6
    ax1.set_ylim(min(0, min(means)) - margin, max(means) + margin)

    # Cumulative line chart
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(opex_cum.index, opex_cum.values, label="OPEX weeks only", color="#27ae60", linewidth=2)
    ax2.plot(bah_cum.index, bah_cum.values, label="Buy & Hold (all weeks)",
             color="#2980b9", linewidth=1.2, alpha=0.75)
    ax2.set_ylabel("Cumulative Return (x)", fontsize=10)
    ax2.set_title(f"{symbol} — Cumulative Return\nOPEX Strategy vs Buy & Hold", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)
    ax2.tick_params(axis="x", labelrotation=30, labelsize=8)

    plt.suptitle(
        "OPEX Effect Backtest  -  Stivers & Sun (2013) Replication",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {out_path}")


# ---------------------------------------------------------------------------
# Per-symbol runner
# ---------------------------------------------------------------------------

def run_backtest(symbol: str, blocked_dates: set) -> dict:
    print(f"\n{'─'*52}")
    print(f"  {symbol}  |  {START_DATE} to {END_DATE}")
    print(f"{'─'*52}")

    df = fetch_bars(symbol, START_DATE, END_DATE)
    print(f"  Daily bars fetched : {len(df)}")

    weekly = compute_weekly_returns(df, blocked_dates)
    opex_rets = weekly[weekly["opex"]]["ret"]
    non_opex_rets = weekly[~weekly["opex"]]["ret"]

    opex_mean = opex_rets.mean() * 100
    non_opex_mean = non_opex_rets.mean() * 100
    opex_sharpe = annualised_sharpe(opex_rets)
    opex_win = win_rate(opex_rets) * 100

    print(f"  OPEX weeks    : {len(opex_rets):>4}  mean={opex_mean:+.4f}%  "
          f"Sharpe={opex_sharpe:.2f}  win={opex_win:.1f}%")
    print(f"  Non-OPEX wks  : {len(non_opex_rets):>4}  mean={non_opex_mean:+.4f}%")

    if symbol == PRIMARY_SYMBOL:
        passed = opex_mean > 0.30 and non_opex_mean < 0.20
        tag = "PASS" if passed else "INFO"
        print(f"  [{tag}] Demo target: OPEX {opex_mean:.3f}% (need >0.30%), "
              f"non-OPEX {non_opex_mean:.3f}% (need <0.20%)")

    return {"symbol": symbol, "weekly": weekly}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("OPEX Backtest - Stivers & Sun (2013) Replication")
    print(f"Symbols: {SYMBOLS}  |  {START_DATE} to {END_DATE}\n")

    print("Loading economic event gate...")
    blocked_dates = fetch_blocked_dates(START_DATE, END_DATE)
    print(f"  Gating event dates : {len(blocked_dates)}")

    out_dir = Path(__file__).parent
    for sym in SYMBOLS:
        result = run_backtest(sym, blocked_dates)
        chart_path = out_dir / f"opex_backtest_{sym}.png"
        plot_results(result["weekly"], sym, chart_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
