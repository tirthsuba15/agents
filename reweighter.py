import json
import time
from math import copysign

from config import HYDRA_DB_SUB_TENANT_ID
import hydradb


_AGENTS = ("sentiment", "momentum", "gamma")
_W_KEYS = [f"w_{a}" for a in _AGENTS]


def _resolve_signals(trade: dict) -> dict:
    signals = trade.get("signals_json", {})
    if isinstance(signals, str):
        try:
            signals = json.loads(signals)
        except (json.JSONDecodeError, TypeError):
            return {}
    return signals if isinstance(signals, dict) else {}


def _actual_sign(trade: dict) -> int:
    direction = trade.get("direction", 0)
    outcome = trade.get("outcome")
    if outcome is None:
        return 0
    actual = direction if outcome else -direction
    return 1 if actual > 0 else (-1 if actual < 0 else 0)


def compute_agent_accuracy(trades: list[dict]) -> dict:
    counts = {k: 0 for k in _W_KEYS}
    correct = {k: 0 for k in _W_KEYS}

    for t in trades:
        actual = _actual_sign(t)
        if actual == 0:
            continue
        signals = _resolve_signals(t)
        for agent in _AGENTS:
            agent_data = signals.get(agent, {})
            agent_dir = agent_data.get("direction") if isinstance(agent_data, dict) else None
            if agent_dir is None or agent_dir == 0:
                continue
            wk = f"w_{agent}"
            counts[wk] += 1
            if (agent_dir > 0 and actual > 0) or (agent_dir < 0 and actual < 0):
                correct[wk] += 1

    return {k: (correct[k] / counts[k] if counts[k] > 0 else 0.5) for k in _W_KEYS}


def attribute_pnl(trades: list[dict]) -> dict:
    totals = {k: 0.0 for k in _W_KEYS}

    for t in trades:
        actual = _actual_sign(t)
        if actual == 0:
            continue
        pnl = t.get("pnl_bps", 0) or 0
        signals = _resolve_signals(t)
        for agent in _AGENTS:
            agent_data = signals.get(agent, {})
            agent_dir = agent_data.get("direction") if isinstance(agent_data, dict) else None
            if agent_dir is None or agent_dir == 0:
                continue
            agent_correct = (agent_dir > 0 and actual > 0) or (agent_dir < 0 and actual < 0)
            totals[f"w_{agent}"] += (1 if agent_correct else -1) * pnl

    return {k: round(v, 2) for k, v in totals.items()}


def reweight(n: int = 50, alpha: float = 0.3, trigger: str = "reweight") -> dict:
    trades = hydradb.get_recent_trades(n)
    completed = [t for t in trades if t.get("outcome") is not None]

    if not completed:
        cur = hydradb.get_agent_weights()
        return {"weights": cur, "accuracy": {}, "retrain": {}, "n_trades": 0}

    acc = compute_agent_accuracy(completed)
    cur = hydradb.get_agent_weights()

    raw = {}
    for k in _W_KEYS:
        raw[k] = alpha * acc[k] + (1 - alpha) * cur.get(k, 1 / 3)
        raw[k] = max(0.05, min(0.80, raw[k]))

    total = sum(raw.values())
    new_weights = {k: v / total for k, v in raw.items()}

    retrain_flags = {
        "momentum": acc["w_momentum"] < 0.52,
        "gamma": acc["w_gamma"] < 0.52,
    }

    pnl_attr = attribute_pnl(completed)
    accuracy_json = {
        **acc,
        "pnl_attribution": pnl_attr,
        "retrain": retrain_flags,
        "n_trades": len(completed),
    }

    hydradb.log_weights(
        new_weights["w_sentiment"],
        new_weights["w_momentum"],
        new_weights["w_gamma"],
        trigger=trigger,
        accuracy_json=accuracy_json,
    )

    return {
        "weights": new_weights,
        "accuracy": acc,
        "retrain": retrain_flags,
        "n_trades": len(completed),
    }


def generate_postmortem(trades: list[dict]) -> str:
    if not trades:
        return "No completed trades to analyze."
    completed = [t for t in trades if t.get("outcome") is not None]
    n = len(completed)
    acc = compute_agent_accuracy(completed)
    pnl = attribute_pnl(completed)
    lines = [
        f"Post-mortem over {n} completed trades.",
        f"Sentiment accuracy: {acc['w_sentiment']:.1%}",
        f"Momentum accuracy: {acc['w_momentum']:.1%}",
        f"Gamma accuracy: {acc['w_gamma']:.1%}",
        f"Ssentiment PnL attribution: {pnl['w_sentiment']:+.0f} bps",
        f"Momentum PnL attribution: {pnl['w_momentum']:+.0f} bps",
        f"Gamma PnL attribution: {pnl['w_gamma']:+.0f} bps",
    ]
    summary = "\n".join(lines)
    # TODO: wire Qwen3 235B via Nebius — replace deterministic summary with LLM-generated meta-reasoning
    return summary


if __name__ == "__main__":
    failures = []

    def check(label: str, ok: bool, detail: str = ""):
        if ok:
            print(f"  PASS  {label}")
        else:
            print(f"  FAIL  {label}  {detail}")
            failures.append(label)

    cur_sub = HYDRA_DB_SUB_TENANT_ID
    print(f"Using sub-tenant: {cur_sub}")

    if cur_sub == "day_trading":
        print("ERROR: refusing to run acceptance test on day_trading sub-tenant")
        import sys
        sys.exit(1)

    try:
        print("Step 1: init_db...")
        check("init_db succeeds", hydradb.init_db())
    except Exception as e:
        check("init_db succeeds", False, str(e))
        import sys
        sys.exit(1)

    try:
        print("Step 2: log initial weights (0.34, 0.33, 0.33)...")
        hydradb.log_weights(0.34, 0.33, 0.33, "init", {})
        init_weights = None
        deadline = time.time() + 30
        while time.time() < deadline:
            init_weights = hydradb.get_agent_weights()
            if abs(init_weights.get("w_sentiment", 0) - 0.34) < 0.01:
                break
            time.sleep(3)
        check("init weights logged", init_weights.get("w_sentiment") == 0.34,
              f"got {init_weights}")
    except Exception as e:
        check("init weights logged", False, str(e))

    try:
        print("Step 3: insert 6 engineered completed trades...")
        trade_specs = [
            # (direction, outcome, pnl, sent_dir, mom_dir, gam_dir)
            (1, True, 50, 1, 1, -1),    # T1: sent✓ mom✓ gam✗
            (1, True, 30, 1, -1, -1),   # T2: sent✓ mom✗ gam✗
            (-1, True, 40, -1, -1, 1),  # T3: sent✓ mom✓ gam✗
            (-1, False, -20, 1, -1, -1),# T4: sent✓ mom✗ gam✗
            (1, False, -10, 1, 1, -1),  # T5: sent✗ mom✗ gam✓
            (1, True, 60, 1, 1, 1),     # T6: sent✓ mom✓ gam✓
        ]
        # Expected: sent=5/6=0.833, mom=3/6=0.5, gam=2/6=0.333

        sids = []
        for i, (direction, outcome, pnl, sd, md, gd) in enumerate(trade_specs):
            trade = {
                "ticker": f"RWT{i+1}",
                "direction": direction,
                "outcome": outcome,
                "pnl_bps": pnl,
                "regime": "test",
                "signals_json": {
                    "sentiment": {"direction": sd},
                    "momentum": {"direction": md},
                    "gamma": {"direction": gd},
                },
                "weights_json": {},
            }
            sid = hydradb.log_trade(trade)
            sids.append(sid)

        print(f"  inserted {len(sids)} trades, waiting for async ingest...")
        for sid in sids:
            hydradb._wait_for(sid, timeout=60, interval=3)
        print("  all trades confirmed ingested")
    except Exception as e:
        import traceback
        check("insert trades", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    try:
        print("Step 4: reweight(n=50)...")
        result = reweight(n=50, alpha=0.3, trigger="acceptance_test")
        w = result["weights"]
        acc = result["accuracy"]
        rt = result["retrain"]

        check("reweight returned weights dict", bool(w))
        check("weights sum ≈ 1.0", abs(sum(w.values()) - 1.0) < 1e-6,
              f"sum={sum(w.values())}")
        for k in ("w_sentiment", "w_momentum", "w_gamma"):
            check(f"{k} in [0.05, 0.80]", 0.05 <= w[k] <= 0.80,
                  f"got {w[k]:.4f}")
        check("w_sentiment > 0.34 (increased from init)",
              w["w_sentiment"] > 0.34,
              f"got {w['w_sentiment']:.4f}")
        check("w_sentiment > w_gamma", w["w_sentiment"] > w["w_gamma"],
              f"{w['w_sentiment']:.4f} <= {w['w_gamma']:.4f}")
        check("retrain.gamma is True (acc < 0.52)", rt["gamma"] is True,
              f"gamma acc={acc['w_gamma']:.4f}")
        check("accuracy entries present", bool(acc.get("w_sentiment")), str(acc))
        check("n_trades >= 6", result["n_trades"] >= 6,
              f"got {result['n_trades']}")

        print()
        print("  Accuracy:", {k: round(v, 4) for k, v in acc.items()})
        print("  New weights:", {k: round(v, 4) for k, v in w.items()})
        print("  Retrain flags:", rt)
    except Exception as e:
        import traceback
        check("reweight", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    try:
        print("Step 5: verify new weights persisted via get_agent_weights...")
        persisted = None
        deadline = time.time() + 30
        while time.time() < deadline:
            persisted = hydradb.get_agent_weights()
            if abs(persisted.get("w_sentiment", 0) - 0.4067) < 0.02:
                break
            time.sleep(3)
        check("persisted w_sentiment ≈ 0.4067",
              abs(persisted.get("w_sentiment", 0) - 0.4067) < 0.02,
              f"got {persisted.get('w_sentiment'):.4f}")
    except Exception as e:
        check("persisted weights", False, str(e))

    try:
        print("Step 6: generate_postmortem stub...")
        trades = hydradb.get_recent_trades(50)
        pm = generate_postmortem(trades)
        check("postmortem returned non-empty string", isinstance(pm, str) and len(pm) > 20)
        lines = pm.split("\n")
        print(f"  first line: {lines[0]}")
    except Exception as e:
        check("generate_postmortem", False, str(e))

    print()
    print("--- CLEANUP ---")
    try:
        print("Clearing all memories from test sub-tenant...")
        deleted = hydradb.clear_all_memories()
        print(f"  deleted {deleted} records")
        time.sleep(3)
        remaining = hydradb._list_all_ids()
        check("sub-tenant is empty after cleanup", len(remaining) == 0,
              f"{len(remaining)} records remain")
    except Exception as e:
        check("cleanup", False, str(e))

    print()
    if failures:
        print(f"RESULT: FAIL  ({len(failures)} check(s) failed)")
    else:
        print("RESULT: PASS  (all checks passed)")
