"""
Thin wrapper over Person C's models/momentum_model.py.
Keeps the graph-facing score_momentum(ticker) signature intact.
Returns a SignalObject dict (direction + conviction + metadata).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from models.momentum_model import get_live_features, predict_momentum
    _REAL = True
except Exception as e:
    print(f"[Momentum] model import failed ({e}), using stub")
    _REAL = False


def score_momentum(ticker: str, price_series: list[float] | None = None) -> dict | float:
    """
    Call Person C's XGBoost momentum model.
    Returns a SignalObject dict on success, falls back to float stub on error.
    """
    if not _REAL:
        print(f"[Momentum STUB] scoring {ticker}")
        return 0.31

    try:
        features = get_live_features(ticker)
        result = predict_momentum(features)
        print(f"[Momentum] {ticker} direction={result['direction']} conviction={result['conviction']}")
        return {
            "agent_id": "momentum",
            "ticker": ticker,
            "direction": result["direction"],
            "conviction": result["conviction"],
            "regime": "momentum" if result["direction"] > 0 else "mean_reversion",
            "horizon_mins": 30,
            "signals": {
                "r1": result.get("r1", 0.0),
                "opex_flag": bool(result.get("opex_flag", 0)),
                "volume_ratio": result.get("features", {}).get("volume_ratio", 1.0),
                "vix_level": result.get("features", {}).get("vix_level", 20.0),
            },
            "model_version": result.get("model", "momentum_xgb_v1"),
        }
    except Exception as e:
        print(f"[Momentum] live inference failed ({e}), using stub")
        return 0.31
