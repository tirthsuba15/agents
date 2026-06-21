import requests

def get_risk_free_rate() -> float:
    """Latest 3M T-bill yield from FRED DGS3MO. Returns decimal (0.05 = 5%)."""
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO", timeout=10)
    r.raise_for_status()
    for line in reversed(r.text.strip().splitlines()):
        _, val = line.split(",")
        if val.strip() not in (".", ""):
            return float(val) / 100
    raise RuntimeError("DGS3MO: no valid data found")
