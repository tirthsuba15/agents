"""
Person B's HydraDB client — stub.
Replace with real implementation when Person B delivers hydradb.py.
Exposes: get_weights, get_agent_weights, query_rag, log_trade, log_pass.
"""
from __future__ import annotations
from typing import Any


def get_weights(ticker: str) -> dict:
    """Fetch signal weights for a ticker. Stub returns defaults."""
    return {
        "w_sentiment": 0.4,
        "w_momentum": 0.35,
        "w_gamma": 0.25,
    }


def get_agent_weights() -> dict:
    """
    Read from agent_weights table (latest row).
    Returns w_sentiment, w_momentum, w_gamma.
    Stub returns equal weights; real impl reads from HydraDB agent_weights table.
    """
    print("[HydraDB STUB] get_agent_weights → defaults")
    return {
        "w_sentiment": 0.33,
        "w_momentum": 0.33,
        "w_gamma": 0.34,
    }


def query_rag(query_text: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve top-k most similar past trade setups by embedding similarity.
    Real impl: embed query_text → vector search → return trade records.
    Stub returns empty list (no past setups in Phase 3).
    """
    print(f"[HydraDB STUB] query_rag top_k={top_k} → no past setups")
    return []


def log_trade(trade_decision: dict, embedding: list[float] | None = None) -> str:
    """Persist trade decision + embedding to HydraDB."""
    print(f"[HydraDB STUB] log_trade: {trade_decision}")
    return "trade_stub_id_001"


def log_pass(reason: str, state_snapshot: dict) -> None:
    """Log a skipped trade (low conviction / regime mismatch / hard gate)."""
    print(f"[HydraDB STUB] log_pass reason={reason} ticker={state_snapshot.get('ticker')}")
