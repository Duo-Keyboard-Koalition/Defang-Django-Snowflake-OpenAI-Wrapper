"""
cortex_rest.py — Snowflake Cortex REST API proxy

Uses Snowflake's inference REST API:
  POST https://<account>.snowflakecomputing.com/api/v2/cortex/inference:complete

This supports:
  - Full OpenAI-compatible messages format
  - Tool / function calling (tools + tool_choice params)
  - Streaming (stream=True)
  - All Cortex models including claude-opus-4-6

Unlike the SQL approach (CORTEX.COMPLETE), the REST API accepts the full
OpenAI request body and returns a fully OpenAI-compatible response — which
means Cursor's agent/edit mode with tool calls works.

Env vars (same as cortex.py):
  SNOWFLAKE_ACCOUNT    e.g. ymuajwd-ym41388
  SNOWFLAKE_USER       e.g. d273liu
  SNOWFLAKE_PASSWORD
  SNOWFLAKE_WAREHOUSE  e.g. COMPUTE_WH
  CORTEX_MODEL         default model
"""

import os
import json
import time
import requests
from requests.auth import HTTPBasicAuth

SNOWFLAKE_ACCOUNT = os.environ.get("SNOWFLAKE_ACCOUNT", "ymuajwd-ym41388")
SNOWFLAKE_USER = os.environ.get("SNOWFLAKE_USER", "d273liu")
SNOWFLAKE_PASSWORD = os.environ.get("SNOWFLAKE_PASSWORD", "")
SNOWFLAKE_WAREHOUSE = os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
DEFAULT_MODEL = os.environ.get("CORTEX_MODEL", "claude-opus-4-6")

# Snowflake Cortex REST inference endpoint
CORTEX_REST_URL = f"https://{SNOWFLAKE_ACCOUNT}.snowflakecomputing.com/api/v2/cortex/inference:complete"


def _get_token() -> str:
    """
    Get a Snowflake JWT or session token for REST API auth.
    Uses basic auth (username:password) via Snowflake's login endpoint to get a token.
    """
    login_url = f"https://{SNOWFLAKE_ACCOUNT}.snowflakecomputing.com/session/v1/login-request"
    resp = requests.post(
        login_url,
        json={
            "data": {
                "CLIENT_APP_ID": "DKKCortexProxy",
                "CLIENT_APP_VERSION": "1.0",
                "SVN_REVISION": "1",
                "ACCOUNT_NAME": SNOWFLAKE_ACCOUNT,
                "LOGIN_NAME": SNOWFLAKE_USER,
                "PASSWORD": SNOWFLAKE_PASSWORD,
                "CLIENT_ENVIRONMENT": {
                    "APPLICATION": "DKKCortexProxy",
                    "OS": "Linux",
                    "OS_VERSION": "1.0",
                    "PYTHON_VERSION": "3.11",
                },
            }
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["data"]["token"]


# Simple in-memory token cache
_token_cache = {"token": None, "expires": 0}


def get_token() -> str:
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires"]:
        return _token_cache["token"]
    token = _get_token()
    _token_cache["token"] = token
    _token_cache["expires"] = now + 3300  # ~55 min
    return token


def complete(
    messages: list,
    model: str = None,
    tools: list = None,
    tool_choice=None,
    stream: bool = False,
    **kwargs,
) -> dict:
    """
    Call Snowflake Cortex REST API with full OpenAI-compatible request.
    Supports tool calling, streaming, all models.

    Returns the raw OpenAI-format response dict.
    """
    _model = model or DEFAULT_MODEL
    token = get_token()

    payload = {
        "model": _model,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if stream:
        payload["stream"] = True

    # Pass through any extra OpenAI params (temperature, max_tokens, etc.)
    for k in ("temperature", "max_tokens", "top_p", "stop"):
        if k in kwargs:
            payload[k] = kwargs[k]

    headers = {
        "Authorization": f'Snowflake Token="{token}"',
        "Content-Type": "application/json",
        "X-Snowflake-Authorization-Token-Type": "SESSION",
        "Accept": "application/json",
    }

    resp = requests.post(
        CORTEX_REST_URL,
        json=payload,
        headers=headers,
        stream=stream,
        timeout=120,
    )

    if not resp.ok:
        raise ValueError(f"Cortex REST error {resp.status_code}: {resp.text}")

    if stream:
        return resp  # caller handles streaming
    return resp.json()
