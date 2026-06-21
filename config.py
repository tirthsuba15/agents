import os


HYDRA_DB_API_KEY = os.environ.get("HYDRA_DB_API_KEY")
HYDRA_DB_TENANT_ID = os.environ.get("HYDRA_DB_TENANT_ID", "agents")
HYDRA_DB_SUB_TENANT_ID = os.environ.get("HYDRA_DB_SUB_TENANT_ID", "day_trading")
HYDRA_DB_BASE_URL = os.environ.get("HYDRA_DB_BASE_URL", "https://api.hydradb.com")

NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY")
NEBIUS_BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com")
MODEL_SENTIMENT = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_META = "Qwen/Qwen3-235B-A22B-Instruct-2507"
MODEL_RESEARCHER = "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1"
