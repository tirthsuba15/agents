"""
StratOS API — read-only FastAPI sidecar for the dashboard UI.

Wraps the existing Python data layer (hydradb, reweighter) and live market
data (Finnhub quotes/news, Alpaca paper account) behind cached JSON endpoints.

DESIGN CONTRACT
  • READ-ONLY. No endpoint runs the LangGraph or submits orders — that would
    place real paper trades and burn Nebius tokens. The dashboard reads
    *persisted* state only (weights, past trades, accuracy) plus live quotes.
  • Every endpoint degrades gracefully: on a missing key / upstream failure it
    returns `{"live": false, ...}` with empty/fallback data instead of 500ing,
    so the UI can show a "demo" badge rather than break.
  • HydraDB list+fetch is multi-second, so everything is TTL-cached in-process.

Run:  ./venv/bin/uvicorn api:app --reload --port 8000
"""
from __future__ import annotations

import json
import math
import statistics
import time
from datetime import datetime, timezone
from typing import Any, Callable

import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

import config
import hydradb
import reweighter

app = FastAPI(title="StratOS API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # dev: vite on :3000 → api on :8000
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────────────
# TTL cache — HydraDB is slow (~4-8s), so never hit it per-request.
# ────────────────────────────────────────────────────────────────
_CACHE: dict[str, tuple[float, Any]] = {}


def cached(key: str, ttl: float, producer: Callable[[], Any]) -> Any:
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    try:
        value = producer()
        _CACHE[key] = (now, value)
        return value
    except Exception as exc:
        # On failure, serve stale if we have it; else re-raise to caller's guard.
        if hit:
            return hit[1]
        raise exc


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ────────────────────────────────────────────────────────────────
# Finnhub (live quotes + news) — plain REST, no SDK dependency.
# ────────────────────────────────────────────────────────────────
_FINNHUB = "https://finnhub.io/api/v1"


def _finnhub_quote(symbol: str) -> dict | None:
    key = config.FINNHUB_API_KEY
    if not key:
        return None
    r = requests.get(f"{_FINNHUB}/quote", params={"symbol": symbol, "token": key}, timeout=8)
    r.raise_for_status()
    d = r.json()
    # Finnhub returns {c,d,dp,h,l,o,pc,t}; all-zero when symbol unknown / closed.
    if not d or d.get("c") in (None, 0):
        return None
    return {
        "symbol": symbol,
        "price": d.get("c"),
        "change": d.get("d"),
        "change_pct": d.get("dp"),
        "high": d.get("h"),
        "low": d.get("l"),
        "open": d.get("o"),
        "prev_close": d.get("pc"),
    }


def _finnhub_news(ticker: str, limit: int = 8) -> list[dict]:
    key = config.FINNHUB_API_KEY
    if not key:
        return []
    today = datetime.now(timezone.utc).date()
    frm = today.replace(day=1) if today.day > 5 else today
    r = requests.get(
        f"{_FINNHUB}/company-news",
        params={"symbol": ticker, "from": str(frm), "to": str(today), "token": key},
        timeout=10,
    )
    r.raise_for_status()
    items = r.json() or []
    out = []
    for n in items[:limit]:
        out.append({
            "source": n.get("source", "—"),
            "headline": n.get("headline", ""),
            "url": n.get("url"),
            "datetime": n.get("datetime"),
            "summary": (n.get("summary") or "")[:200],
        })
    return out


# ────────────────────────────────────────────────────────────────
# Alpaca (paper account + positions) — plain REST, no SDK dependency.
# ────────────────────────────────────────────────────────────────
def _alpaca_headers() -> dict | None:
    if not config.APCA_API_KEY_ID or not config.APCA_API_SECRET_KEY:
        return None
    return {
        "APCA-API-KEY-ID": config.APCA_API_KEY_ID,
        "APCA-API-SECRET-KEY": config.APCA_API_SECRET_KEY,
    }


def _alpaca_get(path: str) -> Any:
    h = _alpaca_headers()
    if h is None:
        return None
    base = config.APCA_BASE_URL.rstrip("/")
    if not base.endswith("/v2"):
        base += "/v2"
    r = requests.get(f"{base}{path}", headers=h, timeout=10)
    r.raise_for_status()
    return r.json()


# ────────────────────────────────────────────────────────────────
# Trade normalization — HydraDB trades are heterogeneous (some are test
# rows missing fields). Normalize to a stable shape for the UI.
# ────────────────────────────────────────────────────────────────
def _parse_ts_from_id(memory_id: str) -> str | None:
    # format: "trade:<iso>:<uuid>" — iso itself contains colons, so rejoin.
    parts = memory_id.split(":")
    if len(parts) >= 3:
        return ":".join(parts[1:-1])
    return None


def _normalize_trade(t: dict) -> dict:
    direction = t.get("direction")
    action = t.get("action")
    if action is None and direction is not None:
        action = "BUY" if direction > 0 else "SELL" if direction < 0 else "PASS"
    return {
        "id": t.get("id"),
        "ticker": t.get("ticker", "—"),
        "action": action or "—",
        "direction": direction,
        "conviction": t.get("conviction"),
        "size_pct": t.get("size_pct"),
        "regime": t.get("regime"),
        "pnl_bps": t.get("pnl_bps"),
        "outcome": t.get("outcome"),
        "rationale": t.get("rationale"),
        "timestamp_entry": t.get("timestamp_entry"),
        "timestamp_exit": t.get("timestamp_exit"),
        "order_id": t.get("alpaca_order_id"),
        "fill_price": (t.get("fill") or {}).get("fill_price") if isinstance(t.get("fill"), dict) else None,
    }


# ────────────────────────────────────────────────────────────────
# Endpoints
# ────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health() -> dict:
    services = {
        "hydradb": bool(config.HYDRA_DB_API_KEY),
        "finnhub": bool(config.FINNHUB_API_KEY),
        "alpaca": bool(config.APCA_API_KEY_ID and config.APCA_API_SECRET_KEY),
        "nebius": bool(config.NEBIUS_API_KEY),
    }
    return {
        "status": "ok",
        "live": any(services.values()),
        "services": services,
        "time": _now_iso(),
    }


@app.get("/api/weights")
def weights() -> dict:
    def produce() -> dict:
        latest = hydradb.get_latest_weights()
        if not latest:
            w = hydradb.get_agent_weights()
            return {"live": False, **w, "trigger": None, "timestamp": None, "accuracy_json": {}}
        return {
            "live": True,
            "w_sentiment": latest.get("w_sentiment"),
            "w_momentum": latest.get("w_momentum"),
            "w_gamma": latest.get("w_gamma"),
            "trigger": latest.get("trigger"),
            "timestamp": latest.get("timestamp"),
            "accuracy_json": latest.get("accuracy_json", {}),
        }
    try:
        return cached("weights", 45, produce)
    except Exception as exc:
        return {"live": False, "error": str(exc)[:200],
                "w_sentiment": 0.4, "w_momentum": 0.35, "w_gamma": 0.25,
                "trigger": None, "timestamp": None, "accuracy_json": {}}


@app.get("/api/accuracy")
def accuracy() -> dict:
    """Per-agent accuracy + pnl attribution + retrain flags from the latest
    reweight record (accuracy_json), with a live recompute fallback."""
    def produce() -> dict:
        latest = hydradb.get_latest_weights()
        acc_json = (latest or {}).get("accuracy_json") or {}
        if acc_json:
            return {"live": True, **acc_json}
        trades = hydradb.get_recent_trades(50)
        completed = [t for t in trades if t.get("outcome") is not None]
        if not completed:
            return {"live": False, "w_sentiment": None, "w_momentum": None,
                    "w_gamma": None, "n_trades": 0}
        acc = reweighter.compute_agent_accuracy(completed)
        pnl = reweighter.attribute_pnl(completed)
        return {"live": True, **acc, "pnl_attribution": pnl, "n_trades": len(completed)}
    try:
        return cached("accuracy", 45, produce)
    except Exception as exc:
        return {"live": False, "error": str(exc)[:200], "n_trades": 0}


@app.get("/api/trades")
def trades(n: int = Query(25, ge=1, le=100)) -> dict:
    def produce() -> dict:
        raw = hydradb.get_recent_trades(n)
        norm = [_normalize_trade(t) for t in raw]
        return {"live": True, "count": len(norm), "trades": norm}
    try:
        return cached(f"trades:{n}", 30, produce)
    except Exception as exc:
        return {"live": False, "error": str(exc)[:200], "count": 0, "trades": []}


@app.get("/api/stats")
def stats() -> dict:
    """Headline metrics for the hero. Each field is computed from real trades
    where possible; `source` flags which are derived vs unavailable."""
    def produce() -> dict:
        raw = hydradb.get_recent_trades(100)
        norm = [_normalize_trade(t) for t in raw]
        completed = [t for t in norm if t.get("outcome") is not None]
        convs = [t["conviction"] for t in norm if isinstance(t.get("conviction"), (int, float))]
        wins = [t for t in completed if t.get("outcome") is True]
        pnls = [t["pnl_bps"] for t in completed if isinstance(t.get("pnl_bps"), (int, float))]

        win_rate = (len(wins) / len(completed)) if completed else None
        avg_conv = (sum(convs) / len(convs)) if convs else None
        # Simple Sharpe-like ratio from per-trade pnl (bps); needs >=2 samples.
        sharpe = None
        if len(pnls) >= 2 and statistics.pstdev(pnls) > 0:
            sharpe = (statistics.mean(pnls) / statistics.pstdev(pnls)) * math.sqrt(len(pnls))

        # Live portfolio equity from Alpaca (read-only).
        equity = None
        buying_power = None
        try:
            acct = _alpaca_get("/account")
            if acct:
                equity = float(acct.get("equity", 0) or 0)
                buying_power = float(acct.get("buying_power", 0) or 0)
        except Exception:
            pass

        return {
            "live": True,
            "n_trades": len(norm),
            "n_completed": len(completed),
            "win_rate": win_rate,
            "avg_conviction": avg_conv,
            "sharpe": sharpe,
            "total_pnl_bps": sum(pnls) if pnls else None,
            "equity": equity,
            "buying_power": buying_power,
        }
    try:
        return cached("stats", 45, produce)
    except Exception as exc:
        return {"live": False, "error": str(exc)[:200], "n_trades": 0,
                "win_rate": None, "avg_conviction": None, "sharpe": None}


@app.get("/api/quotes")
def quotes(symbols: str = Query("SPY,QQQ,NVDA,AAPL,MSFT,TSLA,META,AMZN,GOOGL,AMD")) -> dict:
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()][:20]

    def produce() -> dict:
        out = []
        for s in syms:
            try:
                q = _finnhub_quote(s)
                if q:
                    out.append(q)
            except Exception:
                continue
        return {"live": bool(out), "quotes": out}
    try:
        return cached(f"quotes:{','.join(syms)}", 15, produce)
    except Exception as exc:
        return {"live": False, "error": str(exc)[:200], "quotes": []}


@app.get("/api/news")
def news(ticker: str = Query("NVDA")) -> dict:
    ticker = ticker.strip().upper()

    def produce() -> dict:
        items = _finnhub_news(ticker)
        return {"live": bool(items), "ticker": ticker, "news": items}
    try:
        return cached(f"news:{ticker}", 120, produce)
    except Exception as exc:
        return {"live": False, "error": str(exc)[:200], "ticker": ticker, "news": []}


@app.get("/api/portfolio")
def portfolio() -> dict:
    def produce() -> dict:
        acct = _alpaca_get("/account")
        positions = _alpaca_get("/positions") or []
        if acct is None:
            return {"live": False, "account": None, "positions": []}
        pos = [{
            "ticker": p.get("symbol"),
            "qty": float(p.get("qty", 0) or 0),
            "side": p.get("side"),
            "market_value": float(p.get("market_value", 0) or 0),
            "avg_entry_price": float(p.get("avg_entry_price", 0) or 0),
            "unrealized_pl": float(p.get("unrealized_pl", 0) or 0),
            "unrealized_plpc": float(p.get("unrealized_plpc", 0) or 0),
            "current_price": float(p.get("current_price", 0) or 0),
        } for p in positions]
        return {
            "live": True,
            "account": {
                "equity": float(acct.get("equity", 0) or 0),
                "last_equity": float(acct.get("last_equity", 0) or 0),
                "buying_power": float(acct.get("buying_power", 0) or 0),
                "cash": float(acct.get("cash", 0) or 0),
                "status": acct.get("status"),
            },
            "positions": pos,
        }
    try:
        return cached("portfolio", 30, produce)
    except Exception as exc:
        return {"live": False, "error": str(exc)[:200], "account": None, "positions": []}


@app.get("/api/equity_curve")
def equity_curve(period: str = Query("1M"), timeframe: str = Query("1D")) -> dict:
    """Real paper-account equity history from Alpaca portfolio/history."""
    def produce() -> dict:
        hist = _alpaca_get(f"/account/portfolio/history?period={period}&timeframe={timeframe}")
        if not hist or not hist.get("equity"):
            return {"live": False, "points": []}
        ts = hist.get("timestamp", [])
        eq = hist.get("equity", [])
        points = [
            {"t": int(t), "v": float(v)}
            for t, v in zip(ts, eq)
            if v is not None
        ]
        return {"live": bool(points), "base_value": hist.get("base_value"), "points": points}
    try:
        return cached(f"equity_curve:{period}:{timeframe}", 60, produce)
    except Exception as exc:
        return {"live": False, "error": str(exc)[:200], "points": []}


@app.get("/api/memory")
def memory() -> dict:
    """HydraDB memory summary — counts by kind + recent entries — for the
    MemorySchema panel."""
    def produce() -> dict:
        items = hydradb._list_all_ids()
        ids = [m.get("memory_id", "") for m in items if m.get("memory_id")]
        kinds: dict[str, int] = {}
        for mid in ids:
            kind = mid.split(":", 1)[0] if ":" in mid else "other"
            kinds[kind] = kinds.get(kind, 0) + 1
        recent = []
        for mid in sorted(ids, reverse=True)[:12]:
            recent.append({
                "kind": mid.split(":", 1)[0],
                "timestamp": _parse_ts_from_id(mid),
                "id": mid.split(":")[-1][:8],
            })
        return {"live": True, "total": len(ids), "by_kind": kinds, "recent": recent}
    try:
        return cached("memory", 60, produce)
    except Exception as exc:
        return {"live": False, "error": str(exc)[:200], "total": 0, "by_kind": {}, "recent": []}


@app.get("/api/backtest")
def backtest() -> dict:
    """Real OPEX backtest results from backtest/opex_results.json.

    Generated by `python backtest/opex_backtest.py` (yfinance 2015-2024). Read
    from disk — never recomputed on request (the run downloads 20 symbols)."""
    import os as _os

    def produce() -> dict:
        path = _os.path.join(_os.path.dirname(__file__), "backtest", "opex_results.json")
        if not _os.path.exists(path):
            return {"live": False, "reason": "no results — run python backtest/opex_backtest.py"}
        with open(path) as fh:
            data = json.load(fh)
        data["live"] = True
        return data
    try:
        return cached("backtest", 300, produce)
    except Exception as exc:
        return {"live": False, "error": str(exc)[:200]}


@app.get("/")
def root() -> dict:
    return {"service": "StratOS API", "docs": "/docs", "health": "/api/health"}
