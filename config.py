import os


HYDRA_DB_API_KEY = os.environ.get("HYDRA_DB_API_KEY")
HYDRA_DB_TENANT_ID = os.environ.get("HYDRA_DB_TENANT_ID", "agents")
HYDRA_DB_SUB_TENANT_ID = os.environ.get("HYDRA_DB_SUB_TENANT_ID", "day_trading")
HYDRA_DB_BASE_URL = os.environ.get("HYDRA_DB_BASE_URL", "https://api.hydradb.com")

if not HYDRA_DB_API_KEY:
    raise RuntimeError("HYDRA_DB_API_KEY must be set in environment")
