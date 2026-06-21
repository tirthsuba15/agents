"""
Central config loader. All secrets come from environment / .env file.
Import APCA_KEY_ID / APCA_SECRET_KEY wherever Alpaca auth is needed.
"""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

APCA_API_KEY_ID: str = os.environ.get("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY: str = os.environ.get("APCA_API_SECRET_KEY", "")
APCA_BASE_URL: str = os.environ.get("APCA_BASE_URL", "https://paper-api.alpaca.markets")

NEBIUS_API_KEY: str = os.environ.get("NEBIUS_API_KEY", "")
NEBIUS_SERVERLESS_ENDPOINT: str = os.environ.get(
    "NEBIUS_SERVERLESS_ENDPOINT", "https://api.studio.nebius.ai/v1"
)

FINNHUB_API_KEY: str = os.environ.get("FINNHUB_API_KEY", "")

HYDRA_DB_API_KEY: str = os.environ.get("HYDRA_DB_API_KEY", "")
HYDRA_DB_BASE_URL: str = os.environ.get("HYDRA_DB_BASE_URL", "https://api.hydradb.com")
HYDRA_DB_TENANT_ID: str = os.environ.get("HYDRA_DB_TENANT_ID", "agents")
HYDRA_DB_SUB_TENANT_ID: str = os.environ.get("HYDRA_DB_SUB_TENANT_ID", "day_trading")
