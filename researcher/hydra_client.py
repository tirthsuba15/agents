#!/usr/bin/env python3
"""
researcher/hydra_client.py

Minimal HydraDB REST client for writing strategy candidates.
Table: strategy_candidates
Schema inferred from Phase 5 spec — upsert by strategy name + status.
"""

import datetime
import json
from typing import Optional

import requests

try:
    from config import HYDRA_DB_API_KEY, HYDRA_DB_BASE_URL, HYDRA_DB_TENANT_ID, HYDRA_DB_SUB_TENANT_ID
except ImportError:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    HYDRA_DB_API_KEY      = os.environ.get("HYDRA_DB_API_KEY", "")
    HYDRA_DB_BASE_URL     = os.environ.get("HYDRA_DB_BASE_URL", "https://api.hydradb.com")
    HYDRA_DB_TENANT_ID    = os.environ.get("HYDRA_DB_TENANT_ID", "agents")
    HYDRA_DB_SUB_TENANT_ID = os.environ.get("HYDRA_DB_SUB_TENANT_ID", "day_trading")

TABLE = "strategy_candidates"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {HYDRA_DB_API_KEY}",
        "Content-Type": "application/json",
        "X-Tenant-ID": HYDRA_DB_TENANT_ID,
        "X-Sub-Tenant-ID": HYDRA_DB_SUB_TENANT_ID,
    }


def _base() -> str:
    return HYDRA_DB_BASE_URL.rstrip("/")


def insert_strategy(record: dict) -> Optional[dict]:
    """
    Insert a single strategy candidate record into HydraDB.
    record should have: name, signal_description, entry_rule, exit_rule,
                        estimated_sharpe, data_requirements, status, source
    """
    record.setdefault("created_at", datetime.datetime.utcnow().isoformat())
    record.setdefault("status", "candidate")

    if not HYDRA_DB_API_KEY:
        print(f"  [hydra] HYDRA_DB_API_KEY not set — printing record instead:")
        print(f"    {json.dumps(record, indent=2)}")
        return record

    try:
        url  = f"{_base()}/tables/{TABLE}/rows"
        resp = requests.post(url, json=record, headers=_headers(), timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"  [hydra] insert failed: {exc}")
        print(f"    record: {json.dumps(record)}")
        return None


def insert_many(records: list[dict]) -> int:
    """Insert multiple records. Returns count of successful inserts."""
    ok = 0
    for r in records:
        if insert_strategy(r) is not None:
            ok += 1
    return ok
