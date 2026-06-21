"""Central config — unified (Person A app env vars + Person B HydraDB/Nebius model ids).

All secrets come from the environment / optional .env file. No literal secrets here.
Importing this module never raises on a missing key; modules that need a key
(e.g. hydradb) raise a clear error only when an actual API call is attempted.
"""
from __future__ import annotations

import os

try:  # python-dotenv is optional — config still imports without it
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- Alpaca (paper trading) ---
APCA_API_KEY_ID: str = os.environ.get("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY: str = os.environ.get("APCA_API_SECRET_KEY", "")
APCA_BASE_URL: str = os.environ.get("APCA_BASE_URL", "https://paper-api.alpaca.markets")

# --- Nebius (Token Factory + Serverless Studio) ---
NEBIUS_API_KEY: str = os.environ.get("NEBIUS_API_KEY", "")
NEBIUS_BASE_URL: str = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com")
NEBIUS_SERVERLESS_API_KEY: str = os.environ.get("NEBIUS_SERVERLESS_API_KEY", "")
NEBIUS_SERVERLESS_ENDPOINT: str = os.environ.get(
    "NEBIUS_SERVERLESS_ENDPOINT", "https://api.studio.nebius.ai/v1"
)
MODEL_SENTIMENT: str = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_META: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
MODEL_RESEARCHER: str = "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1"

# --- Finnhub (market data / news) ---
FINNHUB_API_KEY: str = os.environ.get("FINNHUB_API_KEY", "")

# --- HydraDB (memory / RAG) ---
HYDRA_DB_API_KEY: str = os.environ.get("HYDRA_DB_API_KEY", "")
HYDRA_DB_BASE_URL: str = os.environ.get("HYDRA_DB_BASE_URL", "https://api.hydradb.com")
HYDRA_DB_TENANT_ID: str = os.environ.get("HYDRA_DB_TENANT_ID", "agents")
HYDRA_DB_SUB_TENANT_ID: str = os.environ.get("HYDRA_DB_SUB_TENANT_ID", "day_trading")
