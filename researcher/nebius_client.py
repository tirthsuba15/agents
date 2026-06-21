#!/usr/bin/env python3
"""
researcher/nebius_client.py

Nebius Token Factory auth + Nemotron inference client.

Token factory flow:
  POST https://iam.api.nebius.cloud/iam/v1/tokens
  body: {"yandexPassportOauthToken": NEBIUS_API_KEY}
  -> {"iamToken": "...", "expiresAt": "..."}

The iamToken is then used as Bearer auth against NEBIUS_SERVERLESS_ENDPOINT.
Falls back to using NEBIUS_API_KEY directly if token exchange fails
(covers both OAuth-key and direct-API-key configurations).
"""

import time
import sys
from typing import Optional

import requests

try:
    from config import NEBIUS_API_KEY, NEBIUS_SERVERLESS_ENDPOINT
except ImportError:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    NEBIUS_API_KEY          = os.environ.get("NEBIUS_API_KEY", "")
    NEBIUS_SERVERLESS_ENDPOINT = os.environ.get(
        "NEBIUS_SERVERLESS_ENDPOINT", "https://api.studio.nebius.ai/v1"
    )

TOKEN_FACTORY_URL = "https://iam.api.nebius.cloud/iam/v1/tokens"
MODEL_ID          = "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1"

_token_cache: dict = {}   # {token, expires_at}


def _get_iam_token() -> str:
    """
    Resolve auth token for Nebius API.

    Strategy:
    1. If NEBIUS_API_KEY starts with 'v1.' it is a static Nebius API key —
       use it directly as Bearer (no token factory exchange needed).
    2. Otherwise treat it as an OAuth token and exchange via Token Factory
       (POST iam.api.nebius.cloud/iam/v1/tokens) for a short-lived IAM token.
    """
    now = time.time()
    cached = _token_cache.get("token")
    if cached and _token_cache.get("expires_at", 0) - now > 300:
        return cached

    if not NEBIUS_API_KEY:
        raise EnvironmentError("NEBIUS_API_KEY is not set. Add it to .env.")

    # Static API key — use directly, no exchange needed
    if NEBIUS_API_KEY.startswith("v1."):
        _token_cache["token"]      = NEBIUS_API_KEY
        _token_cache["expires_at"] = now + 23 * 3600
        return NEBIUS_API_KEY

    # OAuth token — exchange at token factory
    try:
        resp = requests.post(
            TOKEN_FACTORY_URL,
            json={"yandexPassportOauthToken": NEBIUS_API_KEY},
            timeout=10,
        )
        if resp.ok:
            data = resp.json()
            token = data.get("iamToken", "")
            if token:
                _token_cache["token"]      = token
                _token_cache["expires_at"] = now + 12 * 3600 - 300
                return token
    except Exception as exc:
        print(f"  [nebius] token factory exchange failed: {exc} — using API key directly", file=sys.stderr)

    # Fallback: use key directly
    _token_cache["token"]      = NEBIUS_API_KEY
    _token_cache["expires_at"] = now + 3600
    return NEBIUS_API_KEY


def chat_complete(
    prompt: str,
    system: str = "You are a quantitative finance expert.",
    max_tokens: int = 2048,
    temperature: float = 0.2,
    model: Optional[str] = None,
) -> str:
    """
    Call Nemotron via Nebius Token Factory auth.
    Returns the assistant message content as a string.
    """
    token    = _get_iam_token()
    endpoint = NEBIUS_SERVERLESS_ENDPOINT.rstrip("/")
    url      = f"{endpoint}/chat/completions"
    model_id = model or MODEL_ID

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }

    resp = requests.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
