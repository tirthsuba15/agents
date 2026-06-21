import json
import time
import uuid
from datetime import datetime, timezone

import requests

from config import HYDRA_DB_API_KEY, HYDRA_DB_BASE_URL, HYDRA_DB_TENANT_ID, HYDRA_DB_SUB_TENANT_ID

HEADERS = {
    "Authorization": f"Bearer {HYDRA_DB_API_KEY}",
    "Content-Type": "application/json",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_id(prefix: str) -> str:
    return f"{prefix}:{_now_iso()}:{uuid.uuid4()}"


def _request(method: str, path: str, json_body: dict = None) -> dict:
    url = f"{HYDRA_DB_BASE_URL}{path}"
    resp = requests.request(method, url, headers=HEADERS, json=json_body, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _wait_for(source_id: str, timeout: int = 40, interval: int = 3):
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = _request("POST", "/fetch/content", {
            "tenant_id": HYDRA_DB_TENANT_ID,
            "sub_tenant_id": HYDRA_DB_SUB_TENANT_ID,
            "source_id": source_id,
            "mode": "content",
        })
        if data.get("success") and data.get("content"):
            return data
        time.sleep(interval)
    raise TimeoutError(f"Timed out after {timeout}s waiting for {source_id}")


def _list_all_ids() -> list[dict]:
    items = []
    page = 1
    while True:
        data = _request("POST", "/list/data", {
            "tenant_id": HYDRA_DB_TENANT_ID,
            "sub_tenant_id": HYDRA_DB_SUB_TENANT_ID,
            "kind": "memories",
            "page": page,
        })
        batch = data.get("user_memories", [])
        items.extend(batch)
        pagination = data.get("pagination", {})
        if not pagination.get("has_next"):
            break
        page += 1
    return items


def init_db() -> bool:
    data = _request("POST", "/list/data", {
        "tenant_id": HYDRA_DB_TENANT_ID,
        "sub_tenant_id": HYDRA_DB_SUB_TENANT_ID,
        "kind": "memories",
    })
    assert data.get("success") is True, f"init_db failed: {data}"
    return True


def _add_memory(text: str, source_id: str, title: str):
    _request("POST", "/memories/add_memory", {
        "memories": [{
            "text": text,
            "infer": False,
            "is_markdown": False,
            "source_id": source_id,
            "title": title,
        }],
        "tenant_id": HYDRA_DB_TENANT_ID,
        "sub_tenant_id": HYDRA_DB_SUB_TENANT_ID,
        "upsert": True,
    })


def get_weights(ticker: str) -> dict:
    weights = get_latest_weights()
    if weights:
        return weights
    return {
        "w_sentiment": 0.4,
        "w_momentum": 0.35,
        "w_gamma": 0.25,
    }


def log_trade(trade_decision: dict, embedding: list[float] | None = None) -> str:
    sid = _source_id("trade")
    uid = sid.split(":")[-1]
    trade_decision["id"] = uid
    trade_decision.setdefault("timestamp_entry", _now_iso())
    if embedding is not None:
        trade_decision["embedding"] = embedding
    ticker = trade_decision.get("ticker", "UNKNOWN")
    _add_memory(json.dumps(trade_decision), sid, f"trade {ticker}")
    return sid


def log_pass(reason: str, state_snapshot: dict) -> None:
    record = {
        "type": "pass",
        "reason": reason,
        "state_snapshot": state_snapshot,
        "timestamp": _now_iso(),
    }
    sid = _source_id("pass")
    _add_memory(json.dumps(record), sid, f"pass — {reason}")


def update_trade_outcome(trade_id: str, pnl_bps: float, outcome: bool):
    data = _wait_for(trade_id)
    trade = json.loads(data["content"])
    trade["pnl_bps"] = pnl_bps
    trade["outcome"] = outcome
    trade["timestamp_exit"] = _now_iso()
    _add_memory(json.dumps(trade), trade_id, trade.get("ticker", "trade"))


def get_latest_weights() -> dict:
    items = _list_all_ids()
    weight_ids = [m for m in items if m.get("memory_id", "").startswith("weights:")]
    if not weight_ids:
        return {}
    weight_ids.sort(key=lambda x: x["memory_id"], reverse=True)
    latest_id = weight_ids[0]["memory_id"]
    data = _request("POST", "/fetch/content", {
        "tenant_id": HYDRA_DB_TENANT_ID,
        "sub_tenant_id": HYDRA_DB_SUB_TENANT_ID,
        "source_id": latest_id,
        "mode": "content",
    })
    return json.loads(data.get("content", "{}"))


def log_weights(w_sentiment: float, w_momentum: float, w_gamma: float,
                trigger: str, accuracy_json: dict):
    sid = _source_id("weights")
    record = {
        "id": sid.split(":")[-1],
        "timestamp": _now_iso(),
        "w_sentiment": w_sentiment,
        "w_momentum": w_momentum,
        "w_gamma": w_gamma,
        "trigger": trigger,
        "accuracy_json": accuracy_json,
    }
    _add_memory(json.dumps(record), sid, "agent weights")


def log_strategy_candidate(candidate_dict: dict) -> str:
    sid = _source_id("candidate")
    uid = sid.split(":")[-1]
    candidate_dict["id"] = uid
    candidate_dict.setdefault("timestamp", _now_iso())
    desc = candidate_dict.get("description", "strategy")[:40]
    _add_memory(json.dumps(candidate_dict), sid, f"candidate {desc}")
    return sid


def get_recent_trades(n: int = 50) -> list[dict]:
    items = _list_all_ids()
    trade_ids = [m["memory_id"] for m in items if m.get("memory_id", "").startswith("trade:")]
    trade_ids.sort(reverse=True)
    trade_ids = trade_ids[:n]
    trades = []
    for tid in trade_ids:
        data = _request("POST", "/fetch/content", {
            "tenant_id": HYDRA_DB_TENANT_ID,
            "sub_tenant_id": HYDRA_DB_SUB_TENANT_ID,
            "source_id": tid,
            "mode": "content",
        })
        if data.get("content"):
            trades.append(json.loads(data["content"]))
    return trades


if __name__ == "__main__":
    failures = []

    def check(label: str, ok: bool, detail: str = ""):
        if ok:
            print(f"  PASS  {label}")
        else:
            print(f"  FAIL  {label}  {detail}")
            failures.append(label)

    try:
        print("Step 1: init_db...")
        result = init_db()
        check("init_db returns True", result is True)
    except Exception as e:
        check("init_db returns True", False, str(e))
        print("FATAL: cannot connect — aborting")
        import sys
        sys.exit(1)

    try:
        print("Step 2: log_trade...")
        now_iso = _now_iso()
        dummy_trade = {
            "ticker": "TEST",
            "timestamp_entry": now_iso,
            "timestamp_exit": "",
            "direction": 1,
            "conviction": 0.7,
            "signals_json": json.dumps({"ema_cross": True, "volume_spike": False}),
            "weights_json": json.dumps({"w_sentiment": 0.3, "w_momentum": 0.4, "w_gamma": 0.3}),
            "regime": "bull",
            "pnl_bps": None,
            "outcome": None,
            "alpaca_order_id": "test_order_001",
            "setup_description": "Bullish EMA cross on high volume",
        }
        tid = log_trade(dummy_trade)
        print(f"  trade source_id: {tid}")
        check("log_trade returned non-empty", bool(tid))
    except Exception as e:
        check("log_trade returned non-empty", False, str(e))
        tid = None

    if tid:
        try:
            print("Step 3: _wait_for trade...")
            data = _wait_for(tid)
            check("_wait_for succeeded", True)
        except Exception as e:
            check("_wait_for succeeded", False, str(e))
            data = None

        if data:
            try:
                print("Step 4: round-trip verification...")
                content = json.loads(data["content"])
                ticker_ok = content.get("ticker") == "TEST"
                id_ok = content.get("id") == tid.split(":")[-1]
                check("ticker round-trip", ticker_ok, f"got {content.get('ticker')}")
                check("id round-trip", id_ok, f"got {content.get('id')}")
            except Exception as e:
                check("round-trip verification", False, str(e))

    try:
        print("Step 5: log_weights + get_latest_weights...")
        log_weights(0.5, 0.3, 0.2, "reweight", {"overall": 0.85})
        time.sleep(3)
        weights = {}
        deadline = time.time() + 40
        while time.time() < deadline:
            weights = get_latest_weights()
            if weights.get("trigger") == "reweight":
                break
            time.sleep(3)
        check("log_weights found", bool(weights), f"got {weights}")
        check("weights trigger == reweight", weights.get("trigger") == "reweight",
              f"got {weights.get('trigger')}")
        check("weights w_sentiment == 0.5", weights.get("w_sentiment") == 0.5,
              f"got {weights.get('w_sentiment')}")
    except Exception as e:
        check("log_weights + get_latest_weights", False, str(e))

    if tid:
        try:
            print("Step 6: update_trade_outcome...")
            update_trade_outcome(tid, 42.0, True)
            data = _wait_for(tid)
            content = json.loads(data["content"])
            check("pnl_bps == 42.0", content.get("pnl_bps") == 42.0,
                  f"got {content.get('pnl_bps')}")
            check("outcome == True", content.get("outcome") is True,
                  f"got {content.get('outcome')}")
        except Exception as e:
            check("update_trade_outcome", False, str(e))

    if tid:
        try:
            print("Step 7: get_recent_trades...")
            recent = get_recent_trades(5)
            ids = [t.get("id") for t in recent]
            check("test trade in recent trades", tid.split(":")[-1] in ids,
                  f"ids: {ids}")
        except Exception as e:
            check("test trade in recent trades", False, str(e))

    print()
    print("--- CONTRACT SMOKE TEST ---")
    try:
        w = get_weights("NVDA")
        check("get_weights returns dict", isinstance(w, dict))
        check("get_weights has all 3 w_ keys",
              all(k in w for k in ("w_sentiment", "w_momentum", "w_gamma")))
        dummy = {"ticker": "SMOKE", "direction": -1, "conviction": 0.5}
        sid2 = log_trade(dummy, embedding=[0.1, 0.2, 0.3])
        check("log_trade with embedding returns non-empty id", bool(sid2))
        log_pass("low_conviction", {"ticker": "NVDA", "conviction": 0.2})
        check("log_pass runs without error", True)
    except Exception as e:
        check("contract smoke test", False, str(e))

    print()
    if failures:
        print(f"RESULT: FAIL  ({len(failures)} check(s) failed)")
    else:
        print("RESULT: PASS  (all checks passed)")
