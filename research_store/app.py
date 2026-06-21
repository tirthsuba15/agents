#!/usr/bin/env python3
"""
research_store/app.py

Lightweight FastAPI REST endpoint that replaces HydraDB for local/self-hosted use.
Matches the exact URL structure hydra_client.py already uses:

  POST /tables/{table}/rows        — insert one record
  GET  /tables/{table}/rows        — list all records (optional ?status= filter)
  GET  /tables/{table}/rows/{id}   — get single record
  DELETE /tables/{table}/rows/{id} — delete record

Auth: Bearer token checked against HYDRA_DB_API_KEY env var.
Storage: SQLite at research_store/store.db (auto-created).

To run:
    python research_store/app.py
    # or:
    uvicorn research_store.app:app --host 0.0.0.0 --port 8000

Then set in .env:
    HYDRA_DB_BASE_URL=http://localhost:8000

No changes needed to hydra_client.py — endpoint structure is identical.
"""

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY  = os.environ.get("HYDRA_DB_API_KEY", "")
DB_PATH  = Path(__file__).parent / "store.db"
HOST     = os.environ.get("RESEARCH_STORE_HOST", "0.0.0.0")
PORT     = int(os.environ.get("RESEARCH_STORE_PORT", "8000"))

app = FastAPI(title="StratOS Research Store", version="1.0.0")

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _init_db() -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS rows (
                id         TEXT PRIMARY KEY,
                table_name TEXT NOT NULL,
                tenant     TEXT,
                sub_tenant TEXT,
                data       TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_table ON rows(table_name)")
        con.commit()


@contextmanager
def _db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = json.loads(row["data"])
    d["_id"]         = row["id"]
    d["_table"]      = row["table_name"]
    d["_created_at"] = row["created_at"]
    return d


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _check_auth(authorization: Optional[str]) -> None:
    if not API_KEY:
        return  # no key configured — allow all (dev mode)
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token.strip() != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "db": str(DB_PATH)}


@app.post("/tables/{table}/rows", status_code=201)
async def insert_row(
    table: str,
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_tenant_id: Optional[str]     = Header(default=None),
    x_sub_tenant_id: Optional[str] = Header(default=None),
):
    _check_auth(authorization)
    body = await request.json()
    row_id = str(uuid.uuid4())
    now    = time.time()

    with _db() as con:
        con.execute(
            "INSERT INTO rows (id, table_name, tenant, sub_tenant, data, created_at) VALUES (?,?,?,?,?,?)",
            (row_id, table, x_tenant_id, x_sub_tenant_id, json.dumps(body), now),
        )
        con.commit()

    return {"id": row_id, "table": table, **body}


@app.get("/tables/{table}/rows")
def list_rows(
    table: str,
    status: Optional[str] = None,
    limit: int = 100,
    authorization: Optional[str] = Header(default=None),
    x_tenant_id: Optional[str]   = Header(default=None),
):
    _check_auth(authorization)

    with _db() as con:
        rows = con.execute(
            "SELECT * FROM rows WHERE table_name=? ORDER BY created_at DESC LIMIT ?",
            (table, limit),
        ).fetchall()

    results = [_row_to_dict(r) for r in rows]

    if status:
        results = [r for r in results if r.get("status") == status]

    return {"table": table, "count": len(results), "rows": results}


@app.get("/tables/{table}/rows/{row_id}")
def get_row(
    table: str,
    row_id: str,
    authorization: Optional[str] = Header(default=None),
):
    _check_auth(authorization)

    with _db() as con:
        row = con.execute(
            "SELECT * FROM rows WHERE id=? AND table_name=?", (row_id, table)
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Row {row_id} not found in {table}")

    return _row_to_dict(row)


@app.delete("/tables/{table}/rows/{row_id}", status_code=200)
def delete_row(
    table: str,
    row_id: str,
    authorization: Optional[str] = Header(default=None),
):
    _check_auth(authorization)

    with _db() as con:
        affected = con.execute(
            "DELETE FROM rows WHERE id=? AND table_name=?", (row_id, table)
        ).rowcount
        con.commit()

    if not affected:
        raise HTTPException(status_code=404, detail=f"Row {row_id} not found")

    return {"deleted": row_id}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

_init_db()

if __name__ == "__main__":
    import uvicorn
    print(f"Research Store starting on http://{HOST}:{PORT}")
    print(f"Database: {DB_PATH}")
    print(f"Auth: {'enabled' if API_KEY else 'DISABLED (set HYDRA_DB_API_KEY)'}")
    uvicorn.run(app, host=HOST, port=PORT, reload=False)
