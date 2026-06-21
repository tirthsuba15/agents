"""
Alpaca execution layer — stub for Phase 1.
Phase 3: wire real Alpaca REST/WebSocket API.
"""
from __future__ import annotations


def submit_order(trade_decision: dict) -> dict:
    """
    Submit trade to Alpaca. Returns fill confirmation dict.
    Stub simulates a filled order.
    """
    action = trade_decision.get("action", "BUY")
    ticker = trade_decision.get("ticker", "UNKNOWN")
    qty = trade_decision.get("qty", 1)
    price = trade_decision.get("entry_price", 100.0)

    print(f"[Alpaca STUB] {action} {qty}x {ticker} @ {price}")

    return {
        "order_id": "stub_order_001",
        "status": "filled",
        "ticker": ticker,
        "action": action,
        "qty": qty,
        "fill_price": price,
    }
