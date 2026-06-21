#!/usr/bin/env python3
"""
OPEX Backtest — Stivers & Sun (2013) replication
OPEX weeks (Mon open → 3rd Fri close) vs non-OPEX weeks for AAPL & SPY 2015-2024.
Gate: skip OPEX weeks containing FOMC, CPI, or NFP events (Finnhub calendar).

Required env vars:
    ALPACA_API_KEY, ALPACA_SECRET_KEY  — Alpaca Markets data API
    FINNHUB_API_KEY                    — Finnhub economic calendar

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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")

START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
SYMBOLS = ["AAPL", "SPY"]
PRIMARY_SYMBOL = "AAPL"

ALPACA_DATA_BASE = "https://data.alpaca.markets/v2"
FINNHUB_BASE = "https://finnhub.io/api/v1"

# keywords that identify gating events (case-insensitive substring match)
GATE_KEYWORDS = {"fomc", "federal open market", "cpi", "consumer price", "nonfarm", "nfp"}


# ---------------------------------------------------------------------------
# OPEX helpers
# ---------------------------------------------------------------------------

def third_friday(year: int, month: int) -> datetime.date:
    """Return the date of the third Friday of the given month."""
    c = calendar.monthcalendar(year, month)
    # calendar.FRIDAY == 4; skip weeks where Friday falls outside the month (==0)
    fridays = [week[calendar.FRIDAY] for week in c if week[calendar.FRIDAY] != 0]
    return datetime.date(year, month, fridays[2])


def is_opex_week(date: datetime.date) -> bool:
    """True if date falls Mon–Fri of OPEX week (week containing the 3rd Friday)."""
    tf = third_friday(date.year, date.month)
    monday = tf - datetime.timedelta(days=tf.weekday())  # tf.weekday()==4 (Fri)
    return monday <= date <= tf


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _alpaca_headers() -> dict:
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }


def fetch_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch split-adjusted daily OHLCV bars from Alpaca v2 data API."""
    url = f"{ALPACA_DATA_BASE}/stocks/{symbol}/bars"
    params: dict = {
        "timeframe": "1Day",
        "start": start,
        "end": end,
        "limit": 10000,
        "adjustment": "split",
        "feed": "iex",  # IEX is free-tier; swap to 'sip' with paid subscription
    }
    rows = []
    while True:
        resp = requests.get(url, headers=_alpaca_headers(), params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        rows.extend(data.get("bars", []))
        token = data.get("next_page_token")
        if not token:
            break
        params["page_token"] = token

    if not rows:
        raise RuntimeError(f"No bars returned for {symbol} — check API keys / date range")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["t"]).dt.date
    df = (
        df.rename(columns={"o": "open", "c": "close"})
        .set_index("date")
        .sort_index()[["open", "close"]]
    )
    return df


def fetch_blocked_dates(start: str, end: str) -> set[datetime.date]:
    """
    Return a set of dates that contain FOMC, CPI, or NFP events (Finnhub).
    Gracefully returns empty set if the API call fails.
    """
    if not FINNHUB_API_KEY:
        print("  FINNHUB_API_KEY not set — economic event gate disabled")
        return set()

    # Finnhub free tier limits calendar range; chunk into yearly requests
    blocked: set[datetime.date] = set()
    start_dt = datetime.date.fromisoformat(start)
    end_dt = datetime.date.fromisoformat(end)
    cur = start_dt
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
        except requests.RequestException as exc:
            print(f"  Warning: Finnhub calendar fetch failed for {cur.year}: {exc}")
        cur = datetime.date(cur.year + 1, 1, 1)

    return blocked


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def compute_weekly_returns(df: pd.DataFrame, blocked_dates: set) -> pd.DataFrame:
    """
    Group trading days by ISO year-week.
    Weekly return = (last-day close − first-day open) / first-day open.
    OPEX flag is set on the week's first trading day; cleared if the week
    contains a gating event (FOMC/CPI/NFP).
    """
    opens = df["open"].to_dict()
    closes = df["close"].to_dict()

    weeks: dict = {}
    for d in sorted(df.index):
        key = d.isocalendar()[:2]  # (iso_year, iso_week)
        weeks.setdefault(key, []).append(d)

    records = []
    for key in sorted(weeks):
        days = weeks[key]
        first, last = min(days), max(days)
        if first not in opens or last not in closes:
            continue

        ret = (closes[last] - opens[first]) / opens[first]
        opex = is_opex_week(first)

        # Gate: downgrade to non-OPEX if any macro event falls in the week
        if opex and any(d in blocked_dates for d in days):
            opex = False

        records.append({"week_start": first, "opex": opex, "ret": ret})

    return pd.DataFrame(records).set_index("week_start")


def annualised_sharpe(returns: pd.Series, periods_per_year: int = 52) -> float:
    """Annualised Sharpe ratio (risk-free rate = 0)."""
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

    # — Bar chart: mean weekly return —
    ax1 = fig.add_subplot(gs[0])
    labels = ["OPEX weeks", "Non-OPEX weeks"]
    means = [opex_rets.mean() * 100, non_opex_rets.mean() * 100]
    colors = ["#27ae60", "#e74c3c"]
    bars = ax1.bar(labels, means, color=colors, width=0.45, edgecolor="white", linewidth=1.2)
    ax1.axhline(0, color="#7f8c8d", linewidth=0.8, linestyle="--")
    for bar, val in zip(bars, means):
        ypos = val + 0.008 if val >= 0 else val - 0.018
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            ypos,
            f"{val:+.3f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=bar.get_facecolor(),
        )
    ax1.set_ylabel("Mean Weekly Return (%)", fontsize=10)
    ax1.set_title(f"{symbol} — Mean Weekly Return\n(OPEX vs Non-OPEX, 2015–2024)", fontsize=10)
    margin = max(abs(m) for m in means) * 0.6
    ax1.set_ylim(min(0, min(means)) - margin, max(means) + margin)
    ax1.tick_params(axis="x", labelsize=10)

    # — Line chart: cumulative return —
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(opex_cum.index, opex_cum.values, label="OPEX weeks only", color="#27ae60", linewidth=2)
    ax2.plot(bah_cum.index, bah_cum.values, label="Buy & Hold (all weeks)", color="#2980b9",
             linewidth=1.2, alpha=0.75)
    ax2.set_ylabel("Cumulative Return (×)", fontsize=10)
    ax2.set_title(f"{symbol} — Cumulative Return\nOPEX Strategy vs Buy & Hold", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)
    ax2.tick_params(axis="x", labelrotation=30, labelsize=8)

    plt.suptitle(
        "OPEX Effect Backtest  ·  Stivers & Sun (2013) Replication",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart → {out_path}")


# ---------------------------------------------------------------------------
# Per-symbol runner
# ---------------------------------------------------------------------------

def run_backtest(symbol: str, blocked_dates: set) -> dict:
    print(f"\n{'─'*52}")
    print(f"  {symbol}  |  2015-01-01 → 2024-12-31")
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
        gate = opex_mean > 0.30 and non_opex_mean < 0.20
        status = "PASS" if gate else "INFO"
        print(f"  [{status}] Demo target — OPEX {opex_mean:.3f}% (>0.30%), "
              f"non-OPEX {non_opex_mean:.3f}% (<0.20%)")

    return {"symbol": symbol, "weekly": weekly}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    missing = [k for k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY") if not os.environ.get(k)]
    if missing:
        print(f"Error: missing env vars: {', '.join(missing)}")
        sys.exit(1)

    print("OPEX Backtest — Stivers & Sun (2013) Replication")
    print(f"Symbols: {SYMBOLS}  |  {START_DATE} → {END_DATE}\n")

    print("Fetching economic calendar (Finnhub)…")
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
