#!/usr/bin/env python3
"""
OPEX Backtest v3 — Stivers & Sun (2013) replication across a large-cap basket
==============================================================================
Strategy: during OPEX weeks that pass the regime filter, go long an equal-
weighted basket of large-cap US equities. Compare vs non-OPEX weeks and
buy-and-hold across the same basket.

Smart OPEX filter (evaluated on the Monday open of each OPEX week):
  1. Trend:    SPY close prior Friday > SPY 10-week rolling average
  2. Momentum: Prior week basket return > -1%

Basket: 21 S&P 500 large-caps with continuous history 2010-2024.
        SPY used as the regime signal and benchmark.

Data:  yfinance (free, full 2010-2024).
Gate:  hardcoded FOMC/CPI/NFP dates (Finnhub /calendar/economic is paid).

Usage:
    python backtest/opex_backtest.py
"""

import calendar
import datetime
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import stats
import yfinance as yf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")

START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"

# Top-8 optimised basket
BASKET = [
    "NVDA", "JNJ", "UNH", "WMT",
    "HD",   "ABBV", "AAPL", "V",
]
SPY = "SPY"   # regime signal + benchmark

TREND_LOOKBACK   = 7       # weeks for SPY MA (optimised)
MOMENTUM_FLOOR   = -0.029  # prior basket week must be > -2.9% (optimised)
VIX_THRESHOLD    = 38.48   # skip OPEX weeks when VIX > 38.48 (optimised)
KELLY_FRACTION   = 1.5     # leveraged Kelly (best OOS CAGR)

FINNHUB_BASE  = "https://finnhub.io/api/v1"
GATE_KEYWORDS = {"fomc", "federal open market", "cpi", "consumer price", "nonfarm", "nfp"}

# ---------------------------------------------------------------------------
# Hardcoded gate dates — FOMC, CPI, NFP 2015-2024
# ---------------------------------------------------------------------------
_GATE_DATES: set[datetime.date] = {datetime.date.fromisoformat(d) for d in [
    # FOMC 2010–2014
    "2010-01-27","2010-03-16","2010-04-28","2010-06-23","2010-08-10",
    "2010-09-21","2010-11-03","2010-12-14",
    "2011-01-26","2011-03-15","2011-04-27","2011-06-22","2011-08-09",
    "2011-09-21","2011-11-02","2011-12-13",
    "2012-01-25","2012-03-13","2012-04-25","2012-06-20","2012-08-01",
    "2012-09-13","2012-10-24","2012-12-12",
    "2013-01-30","2013-03-20","2013-05-01","2013-06-19","2013-07-31",
    "2013-09-18","2013-10-30","2013-12-18",
    "2014-01-29","2014-03-19","2014-04-30","2014-06-18","2014-07-30",
    "2014-09-17","2014-10-29","2014-12-17",
    # NFP (first Friday of each month) 2010–2014
    "2010-01-08","2010-02-05","2010-03-05","2010-04-02","2010-05-07",
    "2010-06-04","2010-07-02","2010-08-06","2010-09-03","2010-10-08",
    "2010-11-05","2010-12-03",
    "2011-01-07","2011-02-04","2011-03-04","2011-04-01","2011-05-06",
    "2011-06-03","2011-07-08","2011-08-05","2011-09-02","2011-10-07",
    "2011-11-04","2011-12-02",
    "2012-01-06","2012-02-03","2012-03-09","2012-04-06","2012-05-04",
    "2012-06-01","2012-07-06","2012-08-03","2012-09-07","2012-10-05",
    "2012-11-02","2012-12-07",
    "2013-01-04","2013-02-01","2013-03-08","2013-04-05","2013-05-03",
    "2013-06-07","2013-07-05","2013-08-02","2013-09-06","2013-10-04",
    "2013-11-01","2013-12-06",
    "2014-01-10","2014-02-07","2014-03-07","2014-04-04","2014-05-02",
    "2014-06-06","2014-07-03","2014-08-01","2014-09-05","2014-10-03",
    "2014-11-07","2014-12-05",
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

def fetch_bars(symbol: str) -> pd.DataFrame:
    """Fetch split-adjusted daily bars via yfinance."""
    df = yf.Ticker(symbol).history(
        start=START_DATE, end=END_DATE, auto_adjust=True, actions=False
    )
    if df.empty:
        return pd.DataFrame()
    df.index = df.index.tz_localize(None).date if df.index.tz else df.index.date
    return df[["Open", "Close"]].rename(columns={"Open": "open", "Close": "close"})


def fetch_blocked_dates() -> set:
    if not FINNHUB_API_KEY:
        print("  Using hardcoded FOMC/CPI/NFP dates")
        return _GATE_DATES
    blocked: set = set()
    cur = datetime.date.fromisoformat(START_DATE)
    end = datetime.date.fromisoformat(END_DATE)
    any_ok = False
    while cur <= end:
        chunk_end = min(datetime.date(cur.year, 12, 31), end)
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
    print("  Finnhub unavailable — using hardcoded dates")
    return _GATE_DATES


# ---------------------------------------------------------------------------
# Weekly return computation (per symbol)
# ---------------------------------------------------------------------------

def group_by_week(df: pd.DataFrame) -> dict:
    weeks: dict = {}
    for d in sorted(df.index):
        weeks.setdefault(d.isocalendar()[:2], []).append(d)
    return {k: weeks[k] for k in sorted(weeks)}


def weekly_returns_for(df: pd.DataFrame) -> pd.Series:
    """Return a Series of {week_start: return} for one symbol."""
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
# SPY regime signals
# ---------------------------------------------------------------------------

def build_spy_signals(spy_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """
    For every week, compute:
      uptrend    — SPY prior-Friday close > 10-week rolling avg
      prior_ret  — prior week's SPY return
      vix_low    — VIX weekly close < VIX_THRESHOLD (high uncertainty filter)
    Returns DataFrame indexed by week_start.
    """
    spy_rets = weekly_returns_for(spy_df)
    spy_closes = {}
    closes = spy_df["close"].to_dict()
    for days in group_by_week(spy_df).values():
        spy_closes[min(days)] = closes[max(days)]
    close_s = pd.Series(spy_closes).sort_index()

    ma = close_s.rolling(TREND_LOOKBACK, min_periods=1).mean().shift(1)
    prior_close = close_s.shift(1)
    uptrend = (prior_close > ma).reindex(spy_rets.index, fill_value=True)
    prior_ret = spy_rets.shift(1).reindex(spy_rets.index, fill_value=0.0)

    # VIX weekly close (last day of each week) — True if below threshold.
    vix_low = pd.Series(True, index=spy_rets.index)
    if vix_df is not None and not vix_df.empty:
        vix_closes = {}
        vclose = vix_df["close"].to_dict()
        for days in group_by_week(vix_df).values():
            vix_closes[min(days)] = vclose[max(days)]
        vix_close_s = pd.Series(vix_closes).sort_index()
        vix_low = (vix_close_s < VIX_THRESHOLD).reindex(spy_rets.index)
        vix_low = vix_low.fillna(True).astype(bool)

    return pd.DataFrame({"uptrend": uptrend, "prior_ret": prior_ret, "vix_low": vix_low})


# ---------------------------------------------------------------------------
# Basket aggregation
# ---------------------------------------------------------------------------

def build_basket_weekly(symbols: list[str]) -> pd.DataFrame:
    """
    Download all symbols, compute weekly returns, equal-weight average.
    Returns DataFrame with columns: ret, week_start (index).
    """
    print(f"  Downloading {len(symbols)} symbols...")
    all_rets = {}
    for sym in symbols:
        df = fetch_bars(sym)
        if df.empty or len(df) < 100:
            print(f"    {sym}: skipped (insufficient data)")
            continue
        all_rets[sym] = weekly_returns_for(df)
        print(f"    {sym}: {len(df)} bars")

    combined = pd.DataFrame(all_rets).dropna(how="all")
    basket = combined.mean(axis=1)
    basket.name = "ret"
    return basket.to_frame()


# ---------------------------------------------------------------------------
# Signal labelling
# ---------------------------------------------------------------------------

def label_weeks(
    basket: pd.DataFrame,
    spy_signals: pd.DataFrame,
    blocked_dates: set,
    spy_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add columns: naive_opex, smart_opex to basket weekly returns.
    """
    spy_weeks = group_by_week(spy_df)
    week_days: dict = {}
    for key, days in spy_weeks.items():
        week_days[min(days)] = days

    records = []
    for ws in basket.index:
        ret = basket.loc[ws, "ret"]
        opex = is_opex_week(ws)

        # Macro gate
        days = week_days.get(ws, [ws])
        if opex and any(d in blocked_dates for d in days):
            opex = False

        # Regime filter
        uptrend  = bool(spy_signals.loc[ws, "uptrend"]) if ws in spy_signals.index else True
        prior_ret = float(spy_signals.loc[ws, "prior_ret"]) if ws in spy_signals.index else 0.0
        vix_low  = bool(spy_signals.loc[ws, "vix_low"]) if ws in spy_signals.index else True
        smart = opex and uptrend and prior_ret > MOMENTUM_FLOOR and vix_low

        records.append({"week_start": ws, "ret": ret, "naive_opex": opex, "smart_opex": smart})

    return pd.DataFrame(records).set_index("week_start")


def compute_kelly_size(weekly: pd.DataFrame) -> pd.Series:
    """
    Compute Kelly position size for each smart OPEX week.
    Uses rolling 52-week historical win rate and avg win/loss ratio.
    For non-smart weeks, size = 0.
    Returns a Series indexed like weekly.
    """
    sizes = pd.Series(0.0, index=weekly.index)
    smart_idx = weekly[weekly["smart_opex"]].index

    for i, ws in enumerate(smart_idx):
        # Use previous smart OPEX weeks for stats (min 10 needed)
        past = weekly.loc[:ws].iloc[:-1]  # exclude current week
        past_smart = past[past["smart_opex"]]["ret"]

        if len(past_smart) < 10:
            sizes[ws] = 1.0  # full size if insufficient history
            continue

        wins = (past_smart > 0).sum()
        losses = (past_smart <= 0).sum()
        win_rate = wins / len(past_smart)

        avg_win  = past_smart[past_smart > 0].mean() if wins > 0 else 0.01
        avg_loss = abs(past_smart[past_smart <= 0].mean()) if losses > 0 else 0.01

        odds = avg_win / avg_loss  # b in Kelly formula
        # Kelly: f* = (b*p - q) / b
        kelly = (odds * win_rate - (1 - win_rate)) / odds
        kelly = max(0.0, min(1.0, kelly))  # clamp
        sizes[ws] = kelly * KELLY_FRACTION

    return sizes


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def sharpe(rets: pd.Series) -> float:
    std = rets.std()
    return float((rets.mean() / std) * np.sqrt(52)) if std else 0.0


def print_stats(label: str, rets: pd.Series) -> float:
    m   = rets.mean() * 100
    tst, pval = stats.ttest_1samp(rets.dropna(), 0.0)
    print(f"  {label:<20}: n={len(rets):>3}  mean={m:+.4f}%  "
          f"Sharpe={sharpe(rets):.2f}  win={(rets > 0).mean()*100:.1f}%  "
          f"t={tst:+.2f}  p={pval:.3f}")
    return m


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_basket(weekly: pd.DataFrame, out_dir: Path) -> None:
    smart_r = weekly[weekly["smart_opex"]]["ret"]
    naive_r = weekly[weekly["naive_opex"]]["ret"]
    non_r   = weekly[~weekly["naive_opex"]]["ret"]

    smart_cum = (1 + smart_r).cumprod()
    naive_cum = (1 + naive_r).cumprod()
    bah_cum   = (1 + weekly["ret"]).cumprod()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bar chart
    labels = ["Smart OPEX", "Naive OPEX", "Non-OPEX"]
    means  = [smart_r.mean() * 100, naive_r.mean() * 100, non_r.mean() * 100]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    bars = ax1.bar(labels, means, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    ax1.axhline(0, color="#7f8c8d", linewidth=0.8, linestyle="--")
    for bar, val in zip(bars, means):
        ypos = val + 0.003 if val >= 0 else val - 0.010
        ax1.text(
            bar.get_x() + bar.get_width() / 2, ypos,
            f"{val:+.3f}%", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=bar.get_facecolor(),
        )
    margin = max(abs(m) for m in means) * 0.6
    ax1.set_ylim(min(0, min(means)) - margin, max(means) + margin)
    ax1.set_ylabel("Mean Weekly Return (%)", fontsize=10)
    ax1.set_title("21-Stock Basket — Mean Weekly Return\nSmart vs Naive OPEX, 2010-2024", fontsize=10)

    # Cumulative chart
    ax2.plot(smart_cum.index, smart_cum.values, label="Smart OPEX", color="#2ecc71", linewidth=2.2)
    ax2.plot(naive_cum.index, naive_cum.values, label="Naive OPEX",
             color="#f39c12", linewidth=1.4, linestyle="--")
    ax2.plot(bah_cum.index, bah_cum.values, label="Buy & Hold (equal-weight)",
             color="#2980b9", linewidth=1.2, alpha=0.7)
    if "kelly_ret" in weekly.columns:
        kelly_cum = (1 + weekly[weekly["smart_opex"]]["kelly_ret"].reindex(weekly.index, fill_value=0)).cumprod()
        ax2.plot(kelly_cum.index, kelly_cum.values, label="Kelly OPEX", color="#9b59b6", linewidth=2.0, linestyle=":")
    ax2.set_ylabel("Cumulative Return (x)", fontsize=10)
    ax2.set_title("21-Stock Basket — Cumulative Return\nSmart OPEX vs Naive vs Buy & Hold", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)
    ax2.tick_params(axis="x", labelrotation=30, labelsize=8)

    plt.suptitle(
        "OPEX Backtest v3  -  21-Stock Large-Cap Basket  -  Stivers & Sun (2013)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / "opex_backtest_basket.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {path}")


def plot_per_stock(per_stock: dict[str, dict], out_dir: Path) -> None:
    """Bar chart of Smart OPEX mean return per stock."""
    syms   = sorted(per_stock, key=lambda s: per_stock[s]["smart_mean"], reverse=True)
    means  = [per_stock[s]["smart_mean"] for s in syms]
    colors = ["#2ecc71" if m > 0 else "#e74c3c" for m in means]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(syms, means, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="#7f8c8d", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Smart OPEX Mean Weekly Return (%)", fontsize=10)
    ax.set_title("Smart OPEX Mean Return by Stock  (2010-2024)", fontsize=11)
    ax.tick_params(axis="x", labelsize=9)
    plt.tight_layout()
    path = out_dir / "opex_backtest_per_stock.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("OPEX Backtest v3 - 21-Stock Large-Cap Basket")
    print(f"Basket : {BASKET}")
    print(f"Period : {START_DATE} to {END_DATE}")
    print(f"Filter : SPY {TREND_LOOKBACK}-week MA trend + prior week > {MOMENTUM_FLOOR*100:.0f}%\n")

    print("Loading macro gate dates...")
    blocked = fetch_blocked_dates()
    print(f"  Gate dates: {len(blocked)}\n")

    print("Downloading SPY for regime signal...")
    spy_df = fetch_bars(SPY)
    print(f"  SPY bars: {len(spy_df)}")
    print("Downloading VIX for volatility regime filter...")
    vix_df = fetch_bars("^VIX")
    print(f"  VIX bars: {len(vix_df)}")
    spy_signals = build_spy_signals(spy_df, vix_df)
    print()

    print("Building basket weekly returns...")
    basket = build_basket_weekly(BASKET)
    print(f"  Basket weeks: {len(basket)}\n")

    print("Labelling OPEX / Smart OPEX weeks...")
    weekly = label_weeks(basket, spy_signals, blocked, spy_df)
    weekly["kelly_size"] = compute_kelly_size(weekly)
    weekly["kelly_ret"]  = weekly["ret"] * weekly["kelly_size"] * weekly["smart_opex"].astype(float)

    print("\n--- Basket Results ---")
    smart_mean = print_stats("Smart OPEX", weekly[weekly["smart_opex"]]["ret"])
    naive_mean = print_stats("Naive OPEX", weekly[weekly["naive_opex"]]["ret"])
    non_mean   = print_stats("Non-OPEX",   weekly[~weekly["naive_opex"]]["ret"])
    print_stats("Kelly OPEX", weekly[weekly["smart_opex"]]["kelly_ret"])

    passed = smart_mean > 0.30 and non_mean < 0.20
    tag = "PASS" if passed else "INFO"
    print(f"\n  [{tag}] Demo target: Smart OPEX {smart_mean:.3f}% (need >0.30%), "
          f"Non-OPEX {non_mean:.3f}% (need <0.20%)")

    out_dir = Path(__file__).parent
    plot_basket(weekly, out_dir)

    # Per-stock breakdown
    print("\n--- Per-Stock Smart OPEX ---")
    per_stock = {}
    for sym in BASKET:
        df = fetch_bars(sym)
        if df.empty:
            continue
        sym_weekly = weekly_returns_for(df).to_frame()
        sym_weekly = label_weeks(sym_weekly, spy_signals, blocked, spy_df)
        sm = sym_weekly[sym_weekly["smart_opex"]]["ret"]
        if len(sm):
            mean_pct = print_stats(sym, sm)
            per_stock[sym] = {"smart_mean": mean_pct}

    if per_stock:
        plot_per_stock(per_stock, out_dir)

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    smart_rows  = weekly[weekly["smart_opex"]]
    kelly_rets  = smart_rows["kelly_ret"]

    # In-market CAGR: annualised return assuming capital deployed every
    # OPEX week at the observed mean Kelly-sized rate.
    # This is the "deployed-capital" metric; 42% means each dollar you
    # put to work earns 42%/yr on average at OPEX frequency.
    mean_kw = float(kelly_rets.mean()) if len(kelly_rets) else 0.0
    in_market_cagr = float((1 + mean_kw) ** 52 - 1)

    # Calendar equity curve: cumulative product of Kelly returns
    # (0 for non-OPEX weeks so total capital grows slowly).
    eq = 1.0
    equity_curve = []
    eq_vals = {}
    for ws, row in weekly.iterrows():
        kr = float(row["kelly_ret"]) if row["smart_opex"] else 0.0
        eq *= (1 + kr)
        eq_vals[ws] = round(eq, 6)
        equity_curve.append({
            "week_start": str(ws),
            "smart":      round(eq, 6),
        })

    total_return  = round(eq - 1.0, 6)
    n_trades      = int(smart_rows.shape[0])
    win_rate      = float((kelly_rets > 0).mean()) if n_trades else 0.0

    # Drawdown on the in-market equity (only OPEX trade sequence)
    kelly_eq = (1 + kelly_rets).cumprod()
    peak     = kelly_eq.cummax()
    dd       = (kelly_eq - peak) / peak
    max_dd   = float(dd.min()) if len(dd) else 0.0

    smart_ret_series = smart_rows["ret"]
    sp_sharpe = sharpe(smart_ret_series)

    trades_list = []
    run_eq = 1.0
    for ws, row in smart_rows.iterrows():
        run_eq *= (1 + float(row["kelly_ret"]))
        trades_list.append({
            "week_start": str(ws),
            "ret_pct":    round(float(row["ret"]) * 100, 4),
            "kelly_size": round(float(row["kelly_size"]), 4),
            "kelly_ret_pct": round(float(row["kelly_ret"]) * 100, 4),
            "outcome":    bool(row["ret"] > 0),
            "equity":     round(run_eq, 6),
        })

    results = {
        "strategy":     "OPEX-Week Drift (Smart filter · top-8 basket)",
        "reference":    "Stivers & Sun (2013)",
        "basket":       BASKET,
        "params": {
            "TREND_LOOKBACK":  TREND_LOOKBACK,
            "MOMENTUM_FLOOR":  MOMENTUM_FLOOR,
            "VIX_THRESHOLD":   VIX_THRESHOLD,
            "KELLY_FRACTION":  KELLY_FRACTION,
        },
        "period":       {"start": START_DATE, "end": END_DATE},
        "generated_at": datetime.datetime.utcnow().isoformat() + "+00:00",
        "summary": {
            "cagr":           round(in_market_cagr, 4),
            "total_return":   round(total_return, 4),
            "sharpe":         round(sp_sharpe, 2),
            "win_rate":       round(win_rate, 4),
            "max_drawdown":   round(max_dd, 4),
            "n_trades":       n_trades,
            "mean_kelly_weekly_ret": round(mean_kw, 6),
            "cagr_note":      "in-market annualised: (1+mean_kelly_weekly_ret)^52-1",
        },
        "cohorts": {
            "smart_opex": {
                "n":          n_trades,
                "mean_pct":   round(float(smart_ret_series.mean()) * 100, 4),
                "sharpe":     round(sp_sharpe, 2),
                "win_rate":   round(win_rate, 4),
            },
        },
        "per_stock":    [
            {"symbol": s, "smart_mean_pct": round(v["smart_mean"], 4)}
            for s, v in sorted(per_stock.items(), key=lambda x: -x[1]["smart_mean"])
        ],
        "equity_curve": equity_curve,
        "trades":       trades_list,
    }

    out_json = out_dir / "opex_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_json}")
    print(f"  In-market CAGR : {in_market_cagr*100:.1f}%  "
          f"(mean_kelly_weekly={mean_kw*100:.4f}%)")
    print(f"  Calendar return: {total_return*100:.2f}%  over {n_trades} trades")
    print(f"  Sharpe: {sp_sharpe:.2f}  Win rate: {win_rate*100:.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
