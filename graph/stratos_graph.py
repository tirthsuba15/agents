"""
StratOS — LangGraph StateGraph (Phase 3).

Flow:
  fetch_data → sentiment_node → momentum_node → gamma_node → zero_dte_node
  → meta_node (delegates to agents/meta_agent.py — Qwen3 235B fusion)
  ──┬── (action != PASS AND |C| > 0.35) ──→ execute_node → memory_node
    └── (PASS / low conviction)          ──→ memory_node

Signal format: sentiment + zero_dte → SignalObject dicts.
               momentum + gamma → floats (Person C stubs; become SignalObjects in Phase 4).
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
from agents.sentiment import score_sentiment, SignalObject, _fetch_vix
from agents.momentum_model import score_momentum
from agents.gamma_model import score_gamma
from agents.zero_dte import score_zero_dte
from agents.meta_agent import run_meta_agent, MetaDecision, _CONVICTION_MIN
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


def meta_node(state: AgentState) -> dict:
    """
    Phase 3: delegates to agents/meta_agent.py (Qwen3 235B on Nebius).
    run_meta_agent handles: weights, conviction, hard gates, RAG, model call,
    VRP sizing, HydraDB logging, and Alpaca execution.
    Graph routes to execute_node only if meta_agent already executed — otherwise memory_node.
    """
    ticker = state["ticker"]
    signals = state["signals"]

    vix_val = signals.get("sentiment")
    vix = (
        vix_val.get("signals", {}).get("vix")
        if isinstance(vix_val, dict)
        else None
    )

    decision: MetaDecision = run_meta_agent(ticker, signals, vix=vix)

    return {
        "conviction": decision["conviction"],
        "regime": state["regime"],
        "trade_decision": {
            "ticker": ticker,
            "action": decision["action"],
            "size_pct": decision["size_pct"],
            "conviction": decision["conviction"],
            "rationale": decision["rationale"],
            "hard_gate": decision.get("hard_gate"),
            "fill": decision.get("fill"),   # present if meta_agent already executed
        },
    }


def execute_node(state: AgentState) -> dict:
    """
    Phase 3: meta_agent handles execution internally.
    This node is a no-op pass-through — keeps graph structure intact for Phase 4 hooks.
    """
    print(f"[execute_node] trade already handled by meta_agent")
    return {}


def memory_node(state: AgentState) -> dict:
    """Log final state. meta_agent already called hydradb.log_trade / log_pass."""
    trade = state.get("trade_decision") or {}
    print(f"[memory_node] final action={trade.get('action')} conviction={state.get('conviction', 0):.4f}")
    return {}


# ────────────────────────────────────────────────────────────────
# Conditional edge router
# ────────────────────────────────────────────────────────────────

def should_execute(state: AgentState) -> str:
    """Route: execute if meta_agent issued a trade (action != PASS), else memory."""
    trade = state.get("trade_decision") or {}
    action = trade.get("action", "PASS")
    conviction = state.get("conviction", 0.0)

    if action != "PASS" and abs(conviction) > _CONVICTION_MIN:
        print(f"[router] → execute_node  action={action} C={conviction:.4f}")
        return "execute"

    print(f"[router] → memory_node  action={action} C={conviction:.4f}")
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
    import json as _json

    weights = hydradb.get_weights("NVDA")

    # Scenario A: stub signals → low conviction → PASS
    run_scenario(
        label="Scenario A — stub signals (PASS expected without Nebius key)",
        ticker="NVDA",
        regime="bull",
        weights=weights,
    )

    print()

    # Scenario B: inject pre-computed high-conviction signals directly via meta_agent
    # (bypasses the live-fetch chain; demonstrates E2E execute path)
    print("=" * 60)
    print("Scenario B — pre-computed high-conviction signals (EXECUTE path)")
    print("=" * 60)
    from agents.meta_agent import run_meta_agent
    high_signals = {
        "sentiment": {"agent_id": "sentiment", "ticker": "NVDA",
                      "timestamp": datetime.now(UTC).isoformat(),
                      "direction": 0.72, "conviction": 0.80, "regime": "momentum",
                      "horizon_mins": 30,
                      "signals": {"headline_sentiment": 0.72, "event_risk": False, "vix": 16.5},
                      "model_version": "llama-3.3-70b-fast-v1"},
        "momentum": 0.55,
        "gamma": 0.38,
        "zero_dte": {"agent_id": "zero_dte", "ticker": "NVDA",
                     "timestamp": datetime.now(UTC).isoformat(),
                     "direction": 0.0, "conviction": 0.05, "regime": "neutral",
                     "horizon_mins": 30, "signals": {"stub": True},
                     "model_version": "zero-dte-stub-v1"},
    }
    decision = run_meta_agent("NVDA", high_signals, vix=16.5)
    print("-" * 60)
    print("META DECISION:")
    print(_json.dumps(decision, indent=2, default=str))
    print("=" * 60)
