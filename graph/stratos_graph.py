"""
StratOS — LangGraph StateGraph scaffold (Phase 1).

Flow:
  fetch_data → sentiment_node → momentum_node → gamma_node
  → meta_node ──┬── (|conviction| > 0.35 AND regime agreement) ──→ execute_node → memory_node
                └── (pass)                                       ──→ memory_node
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, UTC
from typing import TypedDict

from langgraph.graph import StateGraph, END

# ── stub dependencies (replaced by real modules in later phases) ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydradb
import embedder
from agents.sentiment import score_sentiment
from agents.momentum_model import score_momentum
from agents.gamma_model import score_gamma
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
    """Run sentiment agent on fetched news."""
    ticker = state["ticker"]
    news = state["signals"].get("_raw_news", [])

    score = score_sentiment(ticker, news)

    return {
        "signals": {
            **state["signals"],
            "sentiment": score,
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
            "gamma": score,
        }
    }


def meta_node(state: AgentState) -> dict:
    """
    Fuse signals + weights → conviction score + trade decision.
    Phase 1: weighted sum. Phase 2: replace inner logic with Qwen3 235B on Nebius.
    """
    signals = state["signals"]
    weights = state["weights"]
    ticker = state["ticker"]
    regime = state["regime"]

    s = signals.get("sentiment", 0.0)
    m = signals.get("momentum", 0.0)
    g = signals.get("gamma", 0.0)

    w_s = weights.get("w_sentiment", 0.33)
    w_m = weights.get("w_momentum", 0.33)
    w_g = weights.get("w_gamma", 0.34)

    conviction = (w_s * s) + (w_m * m) + (w_g * g)

    print(f"[meta_node] conviction={conviction:.4f} regime={regime}")
    print(f"  sentiment={s:.3f}×{w_s} + momentum={m:.3f}×{w_m} + gamma={g:.3f}×{w_g}")

    # Phase 2: send signals + weights to Qwen3 235B via Nebius Serverless AI
    # to generate a structured trade decision with reasoning.
    action = "BUY" if conviction > 0 else "SELL"
    trade_decision = {
        "ticker": ticker,
        "action": action,
        "qty": 10,
        "entry_price": 150.0,   # Phase 2: pull from real market data
        "conviction": conviction,
        "regime": regime,
        "signals": {k: v for k, v in signals.items() if not k.startswith("_")},
    }

    return {
        "conviction": conviction,
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
    g.add_node("meta_node", meta_node)
    g.add_node("execute_node", execute_node)
    g.add_node("memory_node", memory_node)

    g.set_entry_point("fetch_data")
    g.add_edge("fetch_data", "sentiment_node")
    g.add_edge("sentiment_node", "momentum_node")
    g.add_edge("momentum_node", "gamma_node")
    g.add_edge("gamma_node", "meta_node")

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
