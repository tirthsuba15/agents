#!/usr/bin/env python3
"""
Intraday Momentum Backtest -- Gao et al. (2018)
===============================================
Validates the first-half / last-half intraday momentum signal using SPY
30-minute bars from 2020-2024.

Signal:
    r1  = first 30-min bar return (9:30-10:00 am ET)
          = (close_bar1 - prev_close) / prev_close
    r13 = last 30-min bar return (3:30-4:00 pm ET)
          = (close_bar13 - close_bar12) / close_bar12

OLS: r13 ~ r1  via scipy.stats.linregress
Targets: slope > 3.0, R^2 > 0.01

Data source priority:
    1. Alpaca historical API (alpaca-py) -- 30-min bars 2020-2024
       if APCA_API_KEY_ID / APCA_API_SECRET_KEY are set in config.py or env
    2. yfinance 30-min bars (last ~730 days)  -- free fallback
       yfinance limits 30m history to the last 60 days; we use max available.
       Note: with <90 days the R^2 target of 0.01 may not be reached.
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving charts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Credential loading -- try config.py first, then env vars
# ---------------------------------------------------------------------------
APCA_API_KEY_ID = ""
APCA_API_SECRET_KEY = ""
APCA_BASE_URL = "https://paper-api.alpaca.markets"

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

try:
    # config.py lives in the project root (one level up from backtest/)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_BASE_URL  # type: ignore
except ImportError:
    APCA_API_KEY_ID = os.environ.get("APCA_API_KEY_ID", "")
    APCA_API_SECRET_KEY = os.environ.get("APCA_API_SECRET_KEY", "")
    APCA_BASE_URL = os.environ.get("APCA_BASE_URL", "https://paper-api.alpaca.markets")

_USE_ALPACA = bool(APCA_API_KEY_ID and APCA_API_SECRET_KEY)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYMBOL = "SPY"  # keep for backward compat
BASKET = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "JPM",  "JNJ",   "UNH",  "HD",
    "WMT",  "PG",   "BAC",   "MA",   "V",
]
START_IDEAL = "2020-01-01"
END_IDEAL   = "2024-12-31"
CHART_PATH = Path(__file__).resolve().parent / "intraday_momentum_chart.png"

# ---------------------------------------------------------------------------
# Data fetching helpers
# ---------------------------------------------------------------------------

def _fetch_alpaca_30min() -> pd.DataFrame:
    """Return a DataFrame of 30-min OHLCV bars for SPY via alpaca-py."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    import datetime

    client = StockHistoricalDataClient(
        api_key=APCA_API_KEY_ID,
        secret_key=APCA_API_SECRET_KEY,
    )
    request = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame(30, TimeFrameUnit.Minute),
        start=datetime.datetime.strptime(START_IDEAL, "%Y-%m-%d"),
        end=datetime.datetime.strptime(END_IDEAL, "%Y-%m-%d"),
        feed="iex",
        adjustment="all",
    )
    bars = client.get_stock_bars(request)
    df = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(SYMBOL, level="symbol")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    return df


def _fetch_yfinance_30min() -> tuple[pd.DataFrame, str, str]:
    """
    Return (DataFrame, actual_start, actual_end) of 30-min bars via yfinance.

    yfinance limits 30-min data to the last ~60 days from today.
    We request 58 days to stay safely within the window.
    """
    import yfinance as yf
    import datetime

    end_dt   = datetime.date.today()
    start_dt = end_dt - datetime.timedelta(days=58)
    start_str = str(start_dt)
    end_str   = str(end_dt)

    print(f"  [fallback] Downloading SPY 30m bars from yfinance "
          f"({start_str} to {end_str}) ...")
    df = yf.download(
        SYMBOL,
        start=start_str,
        end=end_str,
        interval="30m",
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise ValueError("yfinance returned no 30m data for SPY.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")
    return df, start_str, end_str


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def _compute_signals_from_30min(
    df: pd.DataFrame,
    col_close: str = "close",
) -> pd.DataFrame:
    """
    Compute r1 and r13 from 30-minute bars.

    NYSE regular session has 13 half-hour bars:
        bar 1 : 09:30 - 10:00  -> r1
        bar 12: 15:00 - 15:30
        bar 13: 15:30 - 16:00  -> r13

    r1  = (close_bar1 - prev_close) / prev_close
    r13 = (close_bar13 - close_bar12) / close_bar12
    """
    df = df.copy()
    df = df.between_time("09:30", "15:59")

    days = []
    for date, grp in df.groupby(df.index.date):
        grp = grp.sort_index()
        if len(grp) < 13:
            continue
        bar1  = grp.iloc[0]
        bar12 = grp.iloc[11]
        bar13 = grp.iloc[12]
        days.append({
            "date":        date,
            "close_bar1":  float(bar1[col_close]),
            "close_bar12": float(bar12[col_close]),
            "close_bar13": float(bar13[col_close]),
        })

    if not days:
        raise ValueError("No complete trading days (>=13 bars) found in 30-min data.")

    result = pd.DataFrame(days).set_index("date")
    result["prev_close"] = result["close_bar13"].shift(1)
    result = result.dropna()

    result["r1"]  = (result["close_bar1"]  - result["prev_close"])  / result["prev_close"]
    result["r13"] = (result["close_bar13"] - result["close_bar12"]) / result["close_bar12"]
    return result[["r1", "r13"]]


# ---------------------------------------------------------------------------
# Winsorisation helper
# ---------------------------------------------------------------------------

def _winsorise(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def run_backtest() -> None:
    date_label = "2020-2024"

    # 1. Fetch data ---------------------------------------------------------
    if _USE_ALPACA:
        print("Fetching SPY 30-minute bars from Alpaca ...")
        try:
            raw = _fetch_alpaca_30min()
            signals = _compute_signals_from_30min(raw, col_close="close")
            data_source = "Alpaca 30-min bars (2020-2024)"
        except Exception as exc:
            print(f"  Alpaca fetch failed ({exc}); switching to yfinance 30m fallback.")
            raw, s, e = _fetch_yfinance_30min()
            signals = _compute_signals_from_30min(raw, col_close="Close")
            date_label = f"{s} to {e}"
            data_source = f"yfinance 30-min bars ({date_label}, fallback)"
    else:
        print("Alpaca credentials not found -- using yfinance 30m fallback.")
        raw, s, e = _fetch_yfinance_30min()
        signals = _compute_signals_from_30min(raw, col_close="Close")
        date_label = f"{s} to {e}"
        data_source = f"yfinance 30-min bars ({date_label}, fallback)"

    # 2. Winsorise at 1st / 99th percentile --------------------------------
    signals["r1_w"]  = _winsorise(signals["r1"])
    signals["r13_w"] = _winsorise(signals["r13"])

    # 3. OLS regression -----------------------------------------------------
    x = signals["r1_w"].values
    y = signals["r13_w"].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    n_days = len(signals)

    # 4. Print results ------------------------------------------------------
    target_slope_ok = slope > 3.0
    target_r2_ok    = r_squared > 0.01
    pass_fail = "[PASS]" if (target_slope_ok and target_r2_ok) else "[FAIL]"
    reasons = []
    if not target_slope_ok:
        reasons.append(f"slope {slope:.4f} <= 3.0")
    if not target_r2_ok:
        reasons.append(f"R^2 {r_squared:.4f} <= 0.01")
    detail = " -- " + "; ".join(reasons) if reasons else " Slope and R^2 above targets"

    print()
    print(f"Intraday Momentum Backtest (Gao et al. 2018) -- SPY {date_label}")
    print(f"Data source   : {data_source}")
    print(f"Days analysed : {n_days:,}")
    print(f"Slope (b1)    : {slope:.4f}  (target > 3.0)")
    print(f"R^2           : {r_squared:.4f}  (target > 0.01)")
    print(f"p-value       : {p_value:.6f}")
    print(f"{pass_fail}{detail}")

    # 5. Plot scatter + regression line -------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x * 100, y * 100, alpha=0.30, s=12, color="steelblue", label="Daily obs.")

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = intercept + slope * x_line
    ax.plot(x_line * 100, y_line * 100, color="firebrick", linewidth=1.8,
            label=f"OLS: slope={slope:.2f}, R2={r_squared:.3f}, p={p_value:.4f}")

    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel("r1 -- first 30-min return (%)", fontsize=11)
    ax.set_ylabel("r13 -- last 30-min return (%)", fontsize=11)
    ax.set_title(f"Intraday Momentum (Gao et al. 2018)\nSPY {date_label}", fontsize=13)
    ax.legend(fontsize=9)
    fig.tight_layout()

    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(CHART_PATH), dpi=150)
    plt.close(fig)

    try:
        display_path = CHART_PATH.relative_to(
            Path(__file__).resolve().parent.parent
        )
    except ValueError:
        display_path = CHART_PATH
    print(f"Chart saved   : {display_path}")


def run_basket_backtest() -> list[dict]:
    """
    Run the intraday momentum backtest on each stock in BASKET.
    Uses yfinance 30-min bars (last ~58 days).
    Returns list of result dicts for stocks that PASS (slope > 3.0 AND R^2 > 0.01).
    """
    import datetime

    end_dt   = datetime.date.today()
    start_dt = end_dt - datetime.timedelta(days=58)
    start_str = str(start_dt)
    end_str   = str(end_dt)

    print(f"\nBasket Intraday Momentum Backtest ({start_str} to {end_str})")
    print(f"Basket: {BASKET}")
    print("-" * 65)

    passing = []

    for symbol in BASKET:
        try:
            import yfinance as yf
            df = yf.download(
                symbol,
                start=start_str,
                end=end_str,
                interval="30m",
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                print(f"  {symbol:<6}: no data")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize("America/New_York")
            else:
                df.index = df.index.tz_convert("America/New_York")

            # Try 'Close' (yfinance capitalised)
            col = "Close" if "Close" in df.columns else "close"
            signals = _compute_signals_from_30min(df, col_close=col)

            if len(signals) < 15:
                print(f"  {symbol:<6}: insufficient days ({len(signals)})")
                continue

            signals["r1_w"]  = _winsorise(signals["r1"])
            signals["r13_w"] = _winsorise(signals["r13"])

            x = signals["r1_w"].values
            y = signals["r13_w"].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r_squared = r_value ** 2
            n_days = len(signals)

            slope_ok = slope > 3.0
            r2_ok    = r_squared > 0.01
            result_tag = "PASS" if (slope_ok and r2_ok) else "----"

            print(f"  {symbol:<6}: slope={slope:+6.3f}  R²={r_squared:.4f}  "
                  f"p={p_value:.4f}  n={n_days}  [{result_tag}]")

            if slope_ok and r2_ok:
                passing.append({
                    "symbol":    symbol,
                    "slope":     round(slope, 4),
                    "r_squared": round(r_squared, 4),
                    "p_value":   round(p_value, 6),
                    "n_days":    n_days,
                })

        except Exception as exc:
            print(f"  {symbol:<6}: error — {exc}")

    print(f"\nPassing stocks (slope>3.0, R²>0.01): {len(passing)}/{len(BASKET)}")
    if passing:
        passing.sort(key=lambda x: -x["slope"])
        for r in passing:
            print(f"  {r['symbol']}: slope={r['slope']:.3f}  R²={r['r_squared']:.4f}")

    return passing


if __name__ == "__main__":
    import sys
    if "--basket" in sys.argv:
        run_basket_backtest()
    else:
        run_backtest()
        print()
        run_basket_backtest()
