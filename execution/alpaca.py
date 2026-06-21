"""
Alpaca execution layer — Phase 4.
Uses alpaca-py (official SDK). Paper trading by default.
Keys: APCA_API_KEY_ID, APCA_API_SECRET_KEY (from config.py / .env).
"""
from __future__ import annotations

import json
import sys
import os

# Must remove this file's own directory from sys.path BEFORE importing alpaca-py.
# Python adds the script directory to sys.path[0] at startup; since this file is
# named alpaca.py, doing `import alpaca` would find itself and fail with
# "alpaca is not a package". Removing it lets site-packages win.
_own_dir = os.path.dirname(os.path.abspath(__file__))
while _own_dir in sys.path:
    sys.path.remove(_own_dir)

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError

import hydradb

_PAPER_URL = "https://paper-api.alpaca.markets"

_client: TradingClient | None = None


def _get_client() -> TradingClient:
    global _client
    if _client is None:
        key_id = os.environ.get("APCA_API_KEY_ID") or ""
        secret = os.environ.get("APCA_API_SECRET_KEY") or ""
        _client = TradingClient(
            api_key=key_id,
            secret_key=secret,
            paper=True,                  # always paper; flip to False for live
            url_override=_PAPER_URL,
        )
    return _client


# ────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────

def submit_order(
    ticker: str | dict,
    side: str = "BUY",
    qty: int = 1,
    order_type: str = "market",
    trade_id: str | None = None,
) -> dict:
    """
    Submit an order to Alpaca paper trading.

    Can be called two ways:
      submit_order("AAPL", "BUY", 1)              — new signature (Phase 4)
      submit_order({"ticker": "AAPL", "action": "BUY", "qty": 1})  — legacy dict (Phase 1-3)

    Returns fill confirmation dict including Alpaca order_id.
    Logs order_id back to HydraDB if trade_id provided.
    """
    # Handle legacy dict call from Phase 1-3 meta_agent
    if isinstance(ticker, dict):
        trade_dict = ticker
        ticker = trade_dict.get("ticker", "UNKNOWN")
        side = trade_dict.get("action", trade_dict.get("side", "BUY"))
        qty = int(trade_dict.get("qty", 1))
        order_type = trade_dict.get("order_type", "market")
        trade_id = trade_dict.get("trade_id")

    side_enum = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL

    try:
        client = _get_client()

        if order_type.lower() == "market":
            req = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=side_enum,
                time_in_force=TimeInForce.DAY,
            )
        else:
            raise ValueError(f"Unsupported order_type={order_type}. Use 'market'.")

        order = client.submit_order(req)

        order_id = str(order.id)
        fill_price = float(order.filled_avg_price or 0.0)
        status = str(order.status)

        print(
            f"[Alpaca] {side.upper()} {qty}x {ticker} "
            f"order_id={order_id} status={status} fill_price={fill_price}"
        )

        # Log order_id back to HydraDB
        if trade_id:
            hydradb.update_trade_order_id(trade_id, order_id)

        return {
            "order_id": order_id,
            "status": status,
            "ticker": ticker,
            "action": side.upper(),
            "qty": qty,
            "fill_price": fill_price,
        }

    except APIError as e:
        print(f"[Alpaca] API error: {e}", file=sys.stderr)
        return {
            "order_id": None,
            "status": "error",
            "ticker": ticker,
            "action": side.upper(),
            "qty": qty,
            "fill_price": 0.0,
            "error": str(e),
        }
    except Exception as e:
        print(f"[Alpaca] unexpected error: {e}", file=sys.stderr)
        return {
            "order_id": None,
            "status": "error",
            "ticker": ticker,
            "action": side.upper(),
            "qty": qty,
            "fill_price": 0.0,
            "error": str(e)[:200],
        }


def get_positions() -> list[dict]:
    """
    Return all open positions for demo display.
    Each dict: ticker, qty, market_value, unrealized_pl, avg_entry_price.
    """
    try:
        positions = _get_client().get_all_positions()
        return [
            {
                "ticker": p.symbol,
                "qty": float(p.qty),
                "side": p.side.value if hasattr(p.side, "value") else str(p.side),
                "market_value": float(p.market_value or 0),
                "avg_entry_price": float(p.avg_entry_price or 0),
                "unrealized_pl": float(p.unrealized_pl or 0),
                "unrealized_plpc": float(p.unrealized_plpc or 0),
            }
            for p in positions
        ]
    except APIError as e:
        print(f"[Alpaca] get_positions API error: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"[Alpaca] get_positions error: {e}", file=sys.stderr)
        return []


# ────────────────────────────────────────────────────────────────
# Acceptance test
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    qty = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    print(f"[Alpaca] submitting 1-share market BUY of {ticker} on paper account...")
    result = submit_order(ticker, "BUY", qty)

    print("\nOrder result:")
    print(json.dumps(result, indent=2, default=str))

    print("\nCurrent positions:")
    positions = get_positions()
    if positions:
        print(json.dumps(positions, indent=2, default=str))
    else:
        print("  (none or keys not set)")
