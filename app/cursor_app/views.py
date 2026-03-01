"""
cursor_app/views.py

OpenAI-compatible proxy that gives Cursor full tool calling via Anthropic API.
Snowflake is used as the data backend (RAG over REPO_FILES).

Endpoints:
  POST /cursor/v1/chat/completions  — OpenAI-compat, tool calling via Anthropic
  GET  /cursor/v1/models            — model list
  GET  /cursor/health/              — liveness
"""

import json
import os
import time
import uuid
import logging

import requests
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE = "https://api.anthropic.com/v1"
DEFAULT_MODEL = os.environ.get("CURSOR_MODEL", "claude-opus-4-5")

SUPPORTED_MODELS = [
    "claude-opus-4-5",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
]


def _anthropic_to_openai(ant_resp: dict, model: str) -> dict:
    """Convert Anthropic response → OpenAI format."""
    content_blocks = ant_resp.get("content", [])
    tool_calls = []
    text_parts = []

    for block in content_blocks:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })

    message = {"role": "assistant"}
    if tool_calls:
        message["content"] = None
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"
    else:
        message["content"] = "".join(text_parts)
        finish_reason = ant_resp.get("stop_reason", "stop")
        if finish_reason == "end_turn":
            finish_reason = "stop"

    usage = ant_resp.get("usage", {})
    return {
        "id": ant_resp.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    }


def _openai_tools_to_anthropic(tools: list) -> list:
    """Convert OpenAI tool format → Anthropic tool format."""
    result = []
    for t in tools:
        if t.get("type") == "function":
            fn = t["function"]
            result.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
    return result


def _openai_messages_to_anthropic(messages: list):
    """Split system prompt and convert message history to Anthropic format."""
    system = None
    ant_messages = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            system = content
        elif role == "tool":
            # Tool result — convert to Anthropic format
            ant_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": content,
                }],
            })
        elif role == "assistant" and msg.get("tool_calls"):
            # Assistant tool call — convert to Anthropic format
            blocks = []
            if content:
                blocks.append({"type": "text", "text": content})
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "name": fn.get("name", ""),
                    "input": args,
                })
            ant_messages.append({"role": "assistant", "content": blocks})
        else:
            ant_messages.append({"role": role, "content": content})

    return system, ant_messages


@csrf_exempt
def chat_completions(request):
    if request.method == "OPTIONS":
        return _cors(JsonResponse({}))
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return _cors(JsonResponse({"error": "Invalid JSON"}, status=400))

    if not ANTHROPIC_API_KEY:
        return _cors(JsonResponse({
            "error": {"message": "ANTHROPIC_API_KEY not configured", "type": "auth_error"}
        }, status=401))

    messages = body.get("messages", [])
    model = body.get("model", DEFAULT_MODEL)
    # Map OpenAI model names to Anthropic
    if "gpt" in model or model not in SUPPORTED_MODELS:
        model = DEFAULT_MODEL
    tools = body.get("tools")
    tool_choice = body.get("tool_choice")
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 8192)

    system, ant_messages = _openai_messages_to_anthropic(messages)

    payload = {
        "model": model,
        "messages": ant_messages,
        "max_tokens": max_tokens,
    }
    if system:
        payload["system"] = system
    if tools:
        payload["tools"] = _openai_tools_to_anthropic(tools)
    if tool_choice and tool_choice != "auto":
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            payload["tool_choice"] = {"type": "tool", "name": tool_choice["function"]["name"]}
        elif tool_choice == "none":
            payload["tool_choice"] = {"type": "none"}

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "anthropic-beta": "tools-2024-04-04",
    }

    if stream:
        payload["stream"] = True
        resp = requests.post(f"{ANTHROPIC_BASE}/messages", json=payload, headers=headers, stream=True, timeout=120)
        if not resp.ok:
            return _cors(JsonResponse({"error": {"message": resp.text, "type": "api_error"}}, status=resp.status_code))

        def event_stream():
            for chunk in resp.iter_content(chunk_size=None):
                if chunk:
                    yield chunk

        r = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
        r["Cache-Control"] = "no-cache"
        r["X-Accel-Buffering"] = "no"
        return _cors(r)

    resp = requests.post(f"{ANTHROPIC_BASE}/messages", json=payload, headers=headers, timeout=120)
    if not resp.ok:
        return _cors(JsonResponse({"error": {"message": resp.text, "type": "api_error"}}, status=resp.status_code))

    return _cors(JsonResponse(_anthropic_to_openai(resp.json(), model)))


@require_http_methods(["GET"])
def list_models(request):
    return _cors(JsonResponse({
        "object": "list",
        "data": [{"id": m, "object": "model", "owned_by": "anthropic"} for m in SUPPORTED_MODELS],
    }))


@require_http_methods(["GET"])
def health(request):
    return _cors(JsonResponse({
        "status": "ok",
        "service": "cursor_app",
        "backend": "anthropic",
        "tool_calling": True,
        "key_configured": bool(ANTHROPIC_API_KEY),
    }))


def _add_cors(r):
    r["Access-Control-Allow-Origin"] = "*"
    r["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    r["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return r

def _cors(r):
    return _add_cors(r)
