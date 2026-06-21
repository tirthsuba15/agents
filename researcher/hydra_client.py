#!/usr/bin/env python3
"""
researcher/hydra_client.py

HydraDB memory client for storing strategy candidates.
Uses /memories/add_memory (native HydraDB pattern), NOT /tables/rows.
"""

import datetime
import json
import uuid
from typing import Optional

import requests

try:
    from config import HYDRA_DB_API_KEY, HYDRA_DB_BASE_URL, HYDRA_DB_TENANT_ID, HYDRA_DB_SUB_TENANT_ID
except ImportError:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    HYDRA_DB_API_KEY       = os.environ.get("HYDRA_DB_API_KEY", "")
    HYDRA_DB_BASE_URL      = os.environ.get("HYDRA_DB_BASE_URL", "https://api.hydradb.com")
    HYDRA_DB_TENANT_ID     = os.environ.get("HYDRA_DB_TENANT_ID", "agents")
    HYDRA_DB_SUB_TENANT_ID = os.environ.get("HYDRA_DB_SUB_TENANT_ID", "day_trading")


def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _source_id(prefix: str) -> str:
    return f"{prefix}:{_now_iso()}:{uuid.uuid4()}"


def _base() -> str:
    return HYDRA_DB_BASE_URL.rstrip("/")


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {HYDRA_DB_API_KEY}",
        "Content-Type": "application/json",
    }


def _add_memory(text: str, source_id: str, title: str) -> Optional[dict]:
    """POST a memory to HydraDB /memories/add_memory."""
    if not HYDRA_DB_API_KEY:
        print(f"  [hydra] HYDRA_DB_API_KEY not set — skipping store. title={title}")
        return None

    payload = {
        "memories": [{"text": text, "source_id": source_id, "title": title}],
        "tenant_id":     HYDRA_DB_TENANT_ID,
        "sub_tenant_id": HYDRA_DB_SUB_TENANT_ID,
    }
    try:
        url  = f"{_base()}/memories/add_memory"
        resp = requests.post(url, json=payload, headers=_headers(), timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"  [hydra] add_memory failed: {exc}")
        print(f"    source_id={source_id}, title={title}")
        return None


def insert_strategy(record: dict) -> Optional[str]:
    """
    Store a strategy candidate as a HydraDB memory.
    Returns source_id on success, None on failure.
    """
    record.setdefault("timestamp", _now_iso())
    record.setdefault("status", "candidate")

    sid   = _source_id("candidate")
    uid   = sid.split(":")[-1]
    record["id"] = uid

    desc  = record.get("signal_description") or record.get("name") or "strategy"
    title = f"candidate {desc[:40]}"

    if not HYDRA_DB_API_KEY:
        print(f"  [hydra] HYDRA_DB_API_KEY not set — printing record:")
        print(f"    {json.dumps(record, indent=2)}")
        return sid

    result = _add_memory(json.dumps(record), sid, title)
    if result is not None:
        return sid
    return None


def insert_many(records: list[dict]) -> int:
    """Store multiple strategy candidates. Returns count of successes."""
    ok = 0
    for r in records:
        if insert_strategy(r) is not None:
            ok += 1
    return ok
