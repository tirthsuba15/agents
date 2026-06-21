"""
Thin wrapper over Person C's models/gamma_model.py.
Keeps the graph-facing score_gamma(ticker, options_chain) signature intact.
Returns a SignalObject dict (direction + conviction + metadata).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from models.gamma_model import predict_gamma
    _REAL = True
except Exception as e:
    print(f"[Gamma] model import failed ({e}), using stub")
    _REAL = False


def score_gamma(ticker: str, options_chain: dict | None = None) -> dict | float:
    """
    Call Person C's gamma flow model (cross-sectional options-flow ranking).
    Returns a SignalObject dict on success, falls back to float stub on error.
    """
    if not _REAL:
        print(f"[Gamma STUB] scoring {ticker}")
        return -0.18

    try:
        result = predict_gamma(ticker)
        print(f"[Gamma] {ticker} direction={result['direction']} conviction={result['conviction']}")
        signals = result.get("signals", {})
        return {
            "agent_id": "gamma",
            "ticker": ticker,
            "direction": result["direction"],
            "conviction": result["conviction"],
            "regime": signals.get("gex_regime", "unknown"),
            "horizon_mins": 60,
            "signals": {
                "iv_spread": signals.get("iv_spread", 0.0),
                "smirk": signals.get("smirk", 0.0),
                "pcr": signals.get("pcr", 1.0),
                "gex_regime": signals.get("gex_regime", "unknown"),
                "vix_level": signals.get("vix_level", 20.0),
                "vrp_trade": False,
            },
            "model_version": result.get("model", "gamma_formula_v1"),
        }
    except Exception as e:
        print(f"[Gamma] live inference failed ({e}), using stub")
        return -0.18
