"""
StratOS — LangGraph StateGraph (Phase 2).

Flow:
  fetch_data → sentiment_node → momentum_node → gamma_node → zero_dte_node
  → meta_node ──┬── (|conviction| > 0.35 AND regime agreement) ──→ execute_node → memory_node
                └── (pass)                                       ──→ memory_node

Signal format: sentiment + zero_dte return SignalObject dicts (direction, conviction, regime, ...).
               momentum + gamma return floats (Person C stubs — updated in Phase 3).
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, UTC
from typing import TypedDict

from langgraph.graph import StateGraph, END

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydradb
import embedder
from agents.sentiment import score_sentiment, SignalObject
from agents.momentum_model import score_momentum
from agents.gamma_model import score_gamma
from agents.zero_dte import score_zero_dte
from execution.alpaca import submit_order


# ────────────────────────────────────────────────────────────────
# State schema
# ────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    ticker: str
    timestamp: datetime
    signals: dict          # populated by sub-agents; keys: sentiment, momentum, gamma
    weights: dict          # w_sentiment, w_momentum, w_gamma — from HydraDB
    conviction: float      # computed by meta_node
    trade_decision: dict | None
    regime: str            # e.g. "bull", "bear", "neutral"


# ────────────────────────────────────────────────────────────────
# Node implementations
# ────────────────────────────────────────────────────────────────

def fetch_data(state: AgentState) -> dict:
    """Fetch Finnhub news + options chain. Phase 1: returns stubs."""
    ticker = state["ticker"]
    print(f"[fetch_data] fetching data for {ticker}")

    # Phase 2: replace with real Finnhub client calls
    news_stub = [{"headline": f"{ticker} beats earnings", "sentiment": "positive"}]
    options_chain_stub = {"calls": [], "puts": []}

    return {
        "signals": {
            **state.get("signals", {}),
            "_raw_news": news_stub,
            "_raw_options": options_chain_stub,
        }
    }


def sentiment_node(state: AgentState) -> dict:
    """Run sentiment agent. Stores full SignalObject under signals['sentiment']."""
    ticker = state["ticker"]
    news = state["signals"].get("_raw_news", [])

    signal = score_sentiment(ticker, news)

    return {
        "signals": {
            **state["signals"],
            "sentiment": signal,   # SignalObject dict
        }
    }


def momentum_node(state: AgentState) -> dict:
    """Run Person C's momentum model."""
    ticker = state["ticker"]
    score = score_momentum(ticker)

    return {
        "signals": {
            **state["signals"],
            "momentum": score,
        }
    }


def gamma_node(state: AgentState) -> dict:
    """Run Person C's gamma model."""
    ticker = state["ticker"]
    options_chain = state["signals"].get("_raw_options", {})
    score = score_gamma(ticker, options_chain)

    return {
        "signals": {
            **state["signals"],
            "gamma": score,   # float — Person C stub; becomes SignalObject in Phase 3
        }
    }


def zero_dte_node(state: AgentState) -> dict:
    """Run 0DTE flow agent. Stores SignalObject under signals['zero_dte']."""
    ticker = state["ticker"]
    options_chain = state["signals"].get("_raw_options", {})

    signal = score_zero_dte(ticker, options_chain)

    return {
        "signals": {
            **state["signals"],
            "zero_dte": signal,   # SignalObject dict; weight capped at 5% in meta_node
        }
    }


def _extract_direction(signal: SignalObject | float) -> float:
    """Extract scalar direction from either a SignalObject dict or a raw float."""
    if isinstance(signal, dict):
        return float(signal.get("direction", 0.0))
    return float(signal)


def meta_node(state: AgentState) -> dict:
    """
    Fuse signals + weights → conviction score + trade decision.
    Phase 2: weighted sum with SignalObject support + zero_dte capped at 5%.
    Phase 3: replace fusion logic with Qwen3 235B on Nebius Serverless AI.
    """
    signals = state["signals"]
    weights = state["weights"]
    ticker = state["ticker"]
    regime = state["regime"]

    # Extract direction from each signal (handles both SignalObject and float)
    s = _extract_direction(signals.get("sentiment", 0.0))
    m = _extract_direction(signals.get("momentum", 0.0))
    g = _extract_direction(signals.get("gamma", 0.0))
    z = _extract_direction(signals.get("zero_dte", 0.0))

    # Base weights (sum to 1.0 after zero_dte cap)
    w_s = weights.get("w_sentiment", 0.38)
    w_m = weights.get("w_momentum", 0.33)
    w_g = weights.get("w_gamma", 0.24)
    w_z = 0.05  # zero_dte hard-capped at 5%

    # Renormalize base weights to leave 5% for zero_dte
    base_total = w_s + w_m + w_g
    scale = (1.0 - w_z) / base_total if base_total > 0 else 1.0
    w_s, w_m, w_g = w_s * scale, w_m * scale, w_g * scale

    conviction = (w_s * s) + (w_m * m) + (w_g * g) + (w_z * z)

    # Derive regime from sentiment SignalObject if available, else from state
    sentiment_obj = signals.get("sentiment")
    derived_regime = (
        sentiment_obj.get("regime", regime)
        if isinstance(sentiment_obj, dict)
        else regime
    )

    print(f"[meta_node] conviction={conviction:.4f} regime={derived_regime}")
    print(
        f"  s={s:.3f}×{w_s:.3f} + m={m:.3f}×{w_m:.3f} "
        f"+ g={g:.3f}×{w_g:.3f} + z={z:.3f}×{w_z:.3f}"
    )

    # Phase 3: send signals + weights to Qwen3 235B via Nebius Serverless AI.
    action = "BUY" if conviction > 0 else "SELL"
    trade_decision = {
        "ticker": ticker,
        "action": action,
        "qty": 10,
        "entry_price": 150.0,   # Phase 3: pull from real market data
        "conviction": conviction,
        "regime": derived_regime,
        "signal_summary": {
            "sentiment_direction": s,
            "momentum_direction": m,
            "gamma_direction": g,
            "zero_dte_direction": z,
        },
    }

    return {
        "conviction": conviction,
        "regime": derived_regime,
        "trade_decision": trade_decision,
    }


def execute_node(state: AgentState) -> dict:
    """Submit trade decision to Alpaca."""
    fill = submit_order(state["trade_decision"])
    print(f"[execute_node] fill={fill}")

    return {
        "trade_decision": {
            **state["trade_decision"],
            "fill": fill,
        }
    }


def memory_node(state: AgentState) -> dict:
    """Log trade (or pass) + embedding to HydraDB."""
    trade = state["trade_decision"]
    conviction = state["conviction"]

    if trade and trade.get("fill"):
        emb = embedder.embed(str(trade))
        hydradb.log_trade(trade, embedding=emb)
        print(f"[memory_node] logged executed trade")
    else:
        reason = "low_conviction" if abs(conviction) <= 0.35 else "regime_mismatch"
        hydradb.log_pass(reason, state)
        print(f"[memory_node] logged pass — {reason}")

    return {}


# ────────────────────────────────────────────────────────────────
# Conditional edge router
# ────────────────────────────────────────────────────────────────

CONVICTION_THRESHOLD = 0.35

# Regime agreement: meta_node sets trade direction; if regime doesn't match, skip.
REGIME_DIRECTION = {
    "bull": "BUY",
    "bear": "SELL",
    "neutral": None,   # neutral allows either direction
}


def should_execute(state: AgentState) -> str:
    """
    Route after meta_node.
    Execute if |conviction| > threshold AND regime agrees with trade direction.
    Otherwise log the pass.
    """
    conviction = state["conviction"]
    regime = state["regime"]
    action = state.get("trade_decision", {}).get("action")

    high_conviction = abs(conviction) > CONVICTION_THRESHOLD

    expected_action = REGIME_DIRECTION.get(regime)
    regime_agrees = (expected_action is None) or (expected_action == action)

    if high_conviction and regime_agrees:
        print(f"[router] EXECUTE — conviction={conviction:.4f} regime={regime} action={action}")
        return "execute"

    reason = []
    if not high_conviction:
        reason.append(f"|conviction|={abs(conviction):.4f} ≤ {CONVICTION_THRESHOLD}")
    if not regime_agrees:
        reason.append(f"regime={regime} expects {expected_action}, got {action}")
    print(f"[router] PASS — {'; '.join(reason)}")
    return "pass"


# ────────────────────────────────────────────────────────────────
# Build graph
# ────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("fetch_data", fetch_data)
    g.add_node("sentiment_node", sentiment_node)
    g.add_node("momentum_node", momentum_node)
    g.add_node("gamma_node", gamma_node)
    g.add_node("zero_dte_node", zero_dte_node)
    g.add_node("meta_node", meta_node)
    g.add_node("execute_node", execute_node)
    g.add_node("memory_node", memory_node)

    g.set_entry_point("fetch_data")
    g.add_edge("fetch_data", "sentiment_node")
    g.add_edge("sentiment_node", "momentum_node")
    g.add_edge("momentum_node", "gamma_node")
    g.add_edge("gamma_node", "zero_dte_node")
    g.add_edge("zero_dte_node", "meta_node")

    g.add_conditional_edges(
        "meta_node",
        should_execute,
        {
            "execute": "execute_node",
            "pass": "memory_node",
        },
    )

    g.add_edge("execute_node", "memory_node")
    g.add_edge("memory_node", END)

    return g.compile()


# ────────────────────────────────────────────────────────────────
# Acceptance test entrypoint
# ────────────────────────────────────────────────────────────────

def run_scenario(label: str, ticker: str, regime: str, weights: dict) -> dict:
    import json
    graph = build_graph()
    state: AgentState = {
        "ticker": ticker,
        "timestamp": datetime.now(UTC),
        "signals": {},
        "weights": weights,
        "conviction": 0.0,
        "trade_decision": None,
        "regime": regime,
    }
    print("=" * 60)
    print(f"StratOS Phase 1 — {label}")
    print(f"ticker={ticker} regime={regime} weights={weights}")
    print("=" * 60)
    result = graph.invoke(state)
    print("-" * 60)
    print("TRADE DECISION:")
    decision = result.get("trade_decision")
    if decision:
        print(json.dumps({k: v for k, v in decision.items()}, indent=2, default=str))
    else:
        print("None (pass — no trade)")
    print("=" * 60)
    return result


if __name__ == "__main__":
    weights = hydradb.get_weights("NVDA")

    # Scenario A: normal stub scores → low conviction → PASS path
    run_scenario(
        label="Scenario A — low conviction (PASS path)",
        ticker="NVDA",
        regime="bull",
        weights=weights,
    )

    print()

    # Scenario B: override weights to amplify signals past threshold → EXECUTE path
    high_weight = {"w_sentiment": 0.5, "w_momentum": 0.4, "w_gamma": 0.1}
    # With stubs: 0.42*0.5 + 0.31*0.4 + (-0.18)*0.1 = 0.21 + 0.124 - 0.018 = 0.316 — still low
    # Use extremely high weights to force: sentiment=0.42*0.9=0.378, momentum=0.31*0.1=0.031, gamma=0.0
    execute_weights = {"w_sentiment": 0.9, "w_momentum": 0.1, "w_gamma": 0.0}
    # conviction = 0.42*0.9 + 0.31*0.1 = 0.378 + 0.031 = 0.409 > 0.35 ✓
    run_scenario(
        label="Scenario B — high conviction (EXECUTE path)",
        ticker="NVDA",
        regime="bull",
        weights=execute_weights,
    )
