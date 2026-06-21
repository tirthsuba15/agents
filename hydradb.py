"""
Person B's HydraDB client — stub for Phase 1.
Replace with real implementation when Person B delivers hydradb.py.
"""
from __future__ import annotations
from typing import Any


def get_weights(ticker: str) -> dict:
    """Fetch signal weights from HydraDB. Stub returns defaults."""
    return {
        "w_sentiment": 0.4,
        "w_momentum": 0.35,
        "w_gamma": 0.25,
    }


def log_trade(trade_decision: dict, embedding: list[float] | None = None) -> str:
    """Persist trade decision + embedding to HydraDB. Stub prints and returns fake ID."""
    print(f"[HydraDB STUB] log_trade: {trade_decision}")
    return "trade_stub_id_001"


def log_pass(reason: str, state_snapshot: dict) -> None:
    """Log a skipped trade (low conviction / regime mismatch)."""
    print(f"[HydraDB STUB] log_pass reason={reason} ticker={state_snapshot.get('ticker')}")
