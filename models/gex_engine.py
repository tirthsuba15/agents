#!/usr/bin/env python3
"""
GEX Engine — Phase 2
====================
Deterministic Black-Scholes Greeks: Gamma, Vanna, Charm.
Computes net GEX, zero-gamma flip level, and regime label from an
options chain. Feeds into the Meta-agent as a hard regime filter.

Regime rule:
  positive_gamma  → favour mean-reversion strategies
  negative_gamma  → favour momentum strategies

Data sources:
  Options chain: Finnhub /stock/option-chain (falls back to yfinance)
  Risk-free rate: FRED DGS3MO via data/fred_client.py
  Dividend yield: yfinance ticker.info

Usage:
  python models/gex_engine.py NVDA
  python models/gex_engine.py AAPL
"""

import datetime
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import yfinance as yf
from scipy.stats import norm

# Make data/ importable when run directly from repo root or models/
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.fred_client import get_risk_free_rate

# ---------------------------------------------------------------------------
# #11 — 15-minute options chain cache
# ---------------------------------------------------------------------------
_chain_cache: dict = {}   # {symbol: (fetched_at, chain)}
_CACHE_TTL = 900          # 15 minutes in seconds

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")
FINNHUB_BASE    = "https://finnhub.io/api/v1"
MAX_DTE         = 45   # front-month + next-month only (most gamma-dense)
MIN_IV          = 0.01 # ignore near-zero IV (data artifacts)

# ---------------------------------------------------------------------------
# Type alias — normalised options chain
# Each dict represents one (expiry, strike) pair.
# ---------------------------------------------------------------------------
# {
#   "strike":       float,
#   "expiry":       datetime.date,
#   "call_iv":      float,   # 0.0 if absent
#   "put_iv":       float,
#   "call_oi":      int,
#   "put_oi":       int,
#   "call_volume":  int,
#   "put_volume":   int,
# }
OptionsChain = list[dict]


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _tte(expiry: datetime.date) -> float:
    """Time to expiry in years (returns 0 if expired)."""
    return max((expiry - datetime.date.today()).days / 365.25, 0.0)


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float):
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Black-Scholes gamma per share.
    Formula: e^(-qT) * N'(d1) / (S * sigma * sqrt(T))
    """
    if T <= 0 or sigma < MIN_IV or S <= 0 or K <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return math.exp(-q * T) * norm.pdf(d1) / (S * sigma * math.sqrt(T))


# ---------------------------------------------------------------------------
# Core GEX functions
# ---------------------------------------------------------------------------

def compute_gex(
    options_chain: OptionsChain,
    spot_price: float,
    risk_free_rate: float,
    div_yield: float,
) -> float:
    """
    Aggregate net GEX in dollars.

    Per strike:  GEX_K = gamma_K * OI_K * 100 * S^2 * 0.01
    Calls: positive  (dealers long gamma from short calls)
    Puts:  negative  (dealers short gamma from short puts)
    """
    total = 0.0
    for opt in options_chain:
        T = _tte(opt["expiry"])
        if T <= 0:
            continue
        K = opt["strike"]

        call_iv = opt.get("call_iv", 0.0)
        call_oi = opt.get("call_oi", 0)
        put_iv  = opt.get("put_iv", 0.0)
        put_oi  = opt.get("put_oi", 0)
        call_vol = opt.get("call_volume", 0)
        put_vol  = opt.get("put_volume",  0)
        call_oi_eff = call_oi * (1 + call_vol / max(call_oi, 1))
        put_oi_eff  = put_oi  * (1 + put_vol  / max(put_oi,  1))

        if call_iv >= MIN_IV and call_oi > 0:
            g = bs_gamma(spot_price, K, T, risk_free_rate, div_yield, call_iv)
            total += g * call_oi_eff * 100 * spot_price ** 2 * 0.01

        if put_iv >= MIN_IV and put_oi > 0:
            g = bs_gamma(spot_price, K, T, risk_free_rate, div_yield, put_iv)
            total -= g * put_oi_eff * 100 * spot_price ** 2 * 0.01

    return total


def find_zero_gamma_flip(
    options_chain: OptionsChain,
    spot_price: float,
    r: float,
    q: float,
) -> float:
    """
    Scan a spot grid from spot*0.80 to spot*1.20 (60 points), recompute
    net GEX at each point, find the zero-crossing, and linearly interpolate.
    Returns the gamma-flip price.
    """
    grid = np.linspace(spot_price * 0.80, spot_price * 1.20, 60)
    gex  = np.array([compute_gex(options_chain, s, r, q) for s in grid])

    for i in range(len(gex) - 1):
        if gex[i] * gex[i + 1] <= 0:
            dg = gex[i + 1] - gex[i]
            if dg == 0.0:
                return float(grid[i])
            return float(grid[i] - gex[i] * (grid[i + 1] - grid[i]) / dg)

    # No crossing in grid — return the boundary with lowest absolute GEX
    return float(grid[0] if abs(gex[0]) < abs(gex[-1]) else grid[-1])


def get_gex_regime(net_gex: float, spot: float, zero_gamma_flip: float) -> str:
    """
    'positive_gamma' if spot > flip AND net GEX > 0  → mean-reversion
    'negative_gamma' otherwise                        → momentum
    """
    if spot > zero_gamma_flip and net_gex > 0:
        return "positive_gamma"
    return "negative_gamma"


def compute_vanna_charm(
    options_chain: OptionsChain,
    spot_price: float,
    r: float,
    q: float,
) -> dict:
    """
    OI-weighted net Vanna and Charm across all strikes/expiries.

    Vanna  = -e^(-qT) * N'(d1) * d2 / sigma
             Measures delta sensitivity to implied vol (vol-of-vol risk).

    Charm  (call) = -e^(-qT) * [N'(d1)*(2(r-q)T - d2*sigma*sqrt(T)) /
                                  (2T*sigma*sqrt(T))  -  q*N(d1)]
    Charm  (put)  = -e^(-qT) * [N'(d1)*(2(r-q)T - d2*sigma*sqrt(T)) /
                                  (2T*sigma*sqrt(T))  +  q*N(-d1)]
             Measures delta decay per calendar day (pin-risk indicator).

    Sign convention: calls positive, puts negative (same as GEX).
    Returns {'net_vanna': float, 'net_charm': float}
    """
    net_vanna = 0.0
    net_charm = 0.0

    for opt in options_chain:
        T = _tte(opt["expiry"])
        if T <= 0:
            continue
        K      = opt["strike"]
        sqrt_T = math.sqrt(T)

        def _greek_contrib(iv: float, oi: int, is_put: bool) -> tuple[float, float]:
            if iv < MIN_IV or oi <= 0:
                return 0.0, 0.0
            d1, d2 = _d1_d2(spot_price, K, T, r, q, iv)
            npdf   = norm.pdf(d1)
            eq_T   = math.exp(-q * T)

            vanna = -eq_T * npdf * d2 / iv

            denom = 2 * T * iv * sqrt_T
            if denom == 0:
                charm = 0.0
            elif is_put:
                charm = -eq_T * (npdf * (2*(r-q)*T - d2*iv*sqrt_T) / denom
                                 + q * norm.cdf(-d1))
            else:
                charm = -eq_T * (npdf * (2*(r-q)*T - d2*iv*sqrt_T) / denom
                                 - q * norm.cdf(d1))

            return vanna * oi * 100, charm * oi * 100

        call_iv  = opt.get("call_iv", 0.0)
        call_oi  = opt.get("call_oi", 0)
        call_vol = opt.get("call_volume", 0)
        call_oi_eff = call_oi * (1 + call_vol / max(call_oi, 1))
        v, c = _greek_contrib(call_iv, call_oi_eff, is_put=False)
        net_vanna += v
        net_charm += c

        put_iv  = opt.get("put_iv", 0.0)
        put_oi  = opt.get("put_oi", 0)
        put_vol = opt.get("put_volume", 0)
        put_oi_eff = put_oi * (1 + put_vol / max(put_oi, 1))
        v, c = _greek_contrib(put_iv, put_oi_eff, is_put=True)
        net_vanna -= v   # puts negative
        net_charm -= c

    return {"net_vanna": net_vanna, "net_charm": net_charm}


# ---------------------------------------------------------------------------
# Delta exposure (#12)
# ---------------------------------------------------------------------------

def compute_delta_exposure(
    options_chain: OptionsChain,
    spot_price: float,
    r: float,
    q: float,
) -> float:
    """
    Total dealer delta exposure in dollars.
    Call delta positive (dealers long delta), put delta negative.
    Net positive = dealers net long delta vs market.
    """
    total = 0.0
    for opt in options_chain:
        T = _tte(opt["expiry"])
        if T <= 0:
            continue
        K = opt["strike"]

        call_iv  = opt.get("call_iv", 0.0)
        call_oi  = opt.get("call_oi", 0)
        call_vol = opt.get("call_volume", 0)
        call_oi_eff = call_oi * (1 + call_vol / max(call_oi, 1))
        if call_iv >= MIN_IV and call_oi > 0:
            d1, _ = _d1_d2(spot_price, K, T, r, q, call_iv)
            call_delta = math.exp(-q * T) * norm.cdf(d1)
            total += call_delta * call_oi_eff * 100 * spot_price

        put_iv  = opt.get("put_iv", 0.0)
        put_oi  = opt.get("put_oi", 0)
        put_vol = opt.get("put_volume", 0)
        put_oi_eff = put_oi * (1 + put_vol / max(put_oi, 1))
        if put_iv >= MIN_IV and put_oi > 0:
            d1, _ = _d1_d2(spot_price, K, T, r, q, put_iv)
            put_delta = math.exp(-q * T) * (norm.cdf(d1) - 1)
            total += put_delta * put_oi_eff * 100 * spot_price

    return total


# ---------------------------------------------------------------------------
# #9 — GEX by strike chart
# ---------------------------------------------------------------------------

def plot_gex_by_strike(chain, spot, r, q, symbol, out_path=None):
    import matplotlib.pyplot as plt

    strikes, gex_vals = [], []
    for opt in chain:
        T = _tte(opt["expiry"])
        if T <= 0: continue
        K = opt["strike"]
        call_g = bs_gamma(spot, K, T, r, q, opt.get("call_iv", 0))
        put_g  = bs_gamma(spot, K, T, r, q, opt.get("put_iv",  0))
        net    = (call_g * opt.get("call_oi", 0) - put_g * opt.get("put_oi", 0)) * 100 * spot**2 * 0.01
        strikes.append(K)
        gex_vals.append(net / 1e6)   # in $M

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in gex_vals]
    ax.bar(strikes, gex_vals, color=colors, width=(max(strikes)-min(strikes))/len(strikes)*0.8)
    ax.axvline(spot, color="#2980b9", linewidth=1.5, linestyle="--", label=f"Spot ${spot:.0f}")
    ax.axhline(0,    color="#7f8c8d", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Net GEX ($M)")
    ax.set_title(f"{symbol} — GEX by Strike")
    ax.legend()
    plt.tight_layout()
    path = out_path or Path(f"backtest/gex_by_strike_{symbol}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {path}")


# ---------------------------------------------------------------------------
# Data fetching — Finnhub first, yfinance fallback
# ---------------------------------------------------------------------------

def _parse_finnhub(data: list, today: datetime.date) -> OptionsChain:
    """Normalise Finnhub /stock/option-chain response into OptionsChain."""
    chain: OptionsChain = []
    for expiry_block in data:
        try:
            expiry = datetime.date.fromisoformat(expiry_block["expirationDate"])
        except (KeyError, ValueError):
            continue
        if (expiry - today).days > MAX_DTE or expiry <= today:
            continue

        opts   = expiry_block.get("options", {})
        calls  = {o["strike"]: o for o in opts.get("CALL", [])}
        puts   = {o["strike"]: o for o in opts.get("PUT",  [])}
        strikes = set(calls) | set(puts)

        for K in strikes:
            c = calls.get(K, {})
            p = puts.get(K,  {})
            chain.append({
                "strike":      float(K),
                "expiry":      expiry,
                "call_iv":     float(c.get("impliedVolatility") or 0),
                "put_iv":      float(p.get("impliedVolatility") or 0),
                "call_oi":     int(c.get("openInterest")  or 0),
                "put_oi":      int(p.get("openInterest")  or 0),
                "call_volume": int(c.get("volume")        or 0),
                "put_volume":  int(p.get("volume")        or 0),
            })
    return chain


def _parse_yfinance(ticker: yf.Ticker, today: datetime.date) -> OptionsChain:
    """Normalise yfinance option chain into OptionsChain."""
    import pandas as pd
    chain: OptionsChain = []
    for exp_str in ticker.options:
        try:
            expiry = datetime.date.fromisoformat(exp_str)
        except ValueError:
            continue
        if (expiry - today).days > MAX_DTE or expiry <= today:
            continue
        try:
            oc = ticker.option_chain(exp_str)
        except Exception:
            continue

        c_df = oc.calls[["strike", "impliedVolatility", "openInterest", "volume"]].copy()
        c_df.columns = ["strike", "call_iv", "call_oi", "call_volume"]
        p_df = oc.puts[["strike",  "impliedVolatility", "openInterest", "volume"]].copy()
        p_df.columns = ["strike", "put_iv",  "put_oi",  "put_volume"]

        merged = c_df.merge(p_df, on="strike", how="outer").fillna(0)
        for _, row in merged.iterrows():
            chain.append({
                "strike":      float(row["strike"]),
                "expiry":      expiry,
                "call_iv":     float(row["call_iv"]),
                "put_iv":      float(row["put_iv"]),
                "call_oi":     int(row["call_oi"]),
                "put_oi":      int(row["put_oi"]),
                "call_volume": int(row["call_volume"]),
                "put_volume":  int(row["put_volume"]),
            })
    return chain


def fetch_options_chain(symbol: str) -> OptionsChain:
    """
    Try Finnhub first; fall back to yfinance if key missing or 403.
    Returns normalised OptionsChain (max MAX_DTE days to expiry).
    Caches results for _CACHE_TTL seconds (#11).
    """
    import time
    cached = _chain_cache.get(symbol)
    if cached:
        fetched_at, chain = cached
        if time.time() - fetched_at < _CACHE_TTL:
            print(f"  Options source: cache ({len(chain)} strike rows)")
            return chain

    today = datetime.date.today()

    if FINNHUB_API_KEY:
        try:
            r = requests.get(
                f"{FINNHUB_BASE}/stock/option-chain",
                params={"symbol": symbol, "token": FINNHUB_API_KEY},
                timeout=15,
            )
            r.raise_for_status()
            chain = _parse_finnhub(r.json().get("data", []), today)
            if chain:
                print(f"  Options source: Finnhub ({len(chain)} strike rows)")
                _chain_cache[symbol] = (time.time(), chain)
                return chain
            print("  Finnhub returned empty chain — falling back to yfinance")
        except requests.RequestException as exc:
            print(f"  Finnhub options failed ({exc}) — falling back to yfinance")

    ticker = yf.Ticker(symbol)
    chain  = _parse_yfinance(ticker, today)
    print(f"  Options source: yfinance ({len(chain)} strike rows)")
    _chain_cache[symbol] = (time.time(), chain)
    return chain


def fetch_spot_and_div(symbol: str) -> tuple[float, float]:
    """Return (spot_price, dividend_yield_decimal) via yfinance."""
    info = yf.Ticker(symbol).info
    spot = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
    # Compute yield from annual dividend rate to avoid yfinance scaling issues
    annual_div = float(info.get("dividendRate") or 0)
    div_q = annual_div / spot if spot > 0 else 0.0
    return spot, div_q


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(symbol: str) -> None:
    print(f"\nGEX Engine — {symbol}")
    print("=" * 45)

    print("Fetching risk-free rate (FRED DGS3MO)...")
    r = get_risk_free_rate()
    print(f"  Risk-free rate : {r*100:.3f}%")

    print(f"Fetching spot price and dividend yield ({symbol})...")
    spot, q = fetch_spot_and_div(symbol)
    print(f"  Spot           : ${spot:.2f}")
    print(f"  Dividend yield : {q*100:.3f}%")

    print(f"Fetching options chain ({symbol}, <={MAX_DTE} DTE)...")
    chain = fetch_options_chain(symbol)
    if not chain:
        print("  ERROR: empty options chain — cannot compute GEX")
        sys.exit(1)

    print("\nComputing Greeks...")
    net_gex  = compute_gex(chain, spot, r, q)
    flip     = find_zero_gamma_flip(chain, spot, r, q)
    regime   = get_gex_regime(net_gex, spot, flip)
    vc       = compute_vanna_charm(chain, spot, r, q)
    net_dex  = compute_delta_exposure(chain, spot, r, q)
    plot_gex_by_strike(chain, spot, r, q, symbol)

    print(f"\n{'-'*45}")
    print(f"  Net GEX          : ${net_gex:>15,.0f}")
    print(f"  Net DEX          : ${net_dex:>15,.0f}")
    print(f"  Zero-gamma flip  : ${flip:>10.2f}")
    print(f"  Spot vs flip     :  {'ABOVE' if spot > flip else 'BELOW'} flip by ${abs(spot-flip):.2f}")
    print(f"  Regime           :  {regime}")
    print(f"  Net Vanna        : {vc['net_vanna']:>15,.2f}")
    print(f"  Net Charm        : {vc['net_charm']:>15,.2f}")
    print(f"{'-'*45}")

    regime_msg = {
        "positive_gamma": "Market pinned — dealers BUY dips, SELL rips. Favour mean-reversion.",
        "negative_gamma": "Market unanchored — dealers amplify moves. Favour momentum.",
    }
    print(f"\n  {regime_msg[regime]}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python models/gex_engine.py <SYMBOL>")
        sys.exit(1)
    main(sys.argv[1].upper())
