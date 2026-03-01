"""
snowflake_app/views.py

Full OpenAI-compatible endpoints via Snowflake Cortex REST API.
Supports tool calling, streaming, and all Cortex models.
Cursor and any OpenAI SDK client can point here and get full agent/edit support.

Endpoints:
  POST /snowflake/v1/chat/completions  — OpenAI-compat completions (tools supported)
  GET  /snowflake/v1/models            — model list
  POST /snowflake/ask/                 — simple prompt shorthand
  GET  /snowflake/health/              — liveness
"""

import json
import os
import logging
import time
import uuid

from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = [
    "claude-opus-4-6",
    "claude-opus-4-5",
    "snowflake-arctic",
    "mistral-large2",
    "mistral-large",
    "mistral-7b",
    "mixtral-8x7b",
    "llama3.1-70b",
    "llama3.1-8b",
    "llama3-70b",
    "llama3-8b",
    "gemma-7b",
    "reka-flash",
    "reka-core",
    "jamba-instruct",
    "jamba-1.5-mini",
    "jamba-1.5-large",
]


def _import_rest():
    from cortex_app.cortex_rest import complete
    return complete


@csrf_exempt
@require_http_methods(["POST"])
def chat_completions(request):
    """
    POST /snowflake/v1/chat/completions
    Full OpenAI-compatible — passes tools/tool_choice through to Cortex REST API.
    Cursor agent mode works here.
    """
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}, status=400)

    messages = body.get("messages", [])
    if not messages:
        return JsonResponse({"error": {"message": "'messages' required", "type": "invalid_request_error"}}, status=400)

    model = body.get("model") or os.environ.get("CORTEX_MODEL", "claude-opus-4-6")
    tools = body.get("tools")
    tool_choice = body.get("tool_choice")
    stream = body.get("stream", False)
    extra = {k: body[k] for k in ("temperature", "max_tokens", "top_p", "stop") if k in body}

    try:
        complete = _import_rest()

        if stream:
            raw_resp = complete(messages, model=model, tools=tools, tool_choice=tool_choice, stream=True, **extra)

            def event_stream():
                for chunk in raw_resp.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk

            return StreamingHttpResponse(event_stream(), content_type="text/event-stream")

        result = complete(messages, model=model, tools=tools, tool_choice=tool_choice, **extra)
        return JsonResponse(result)

    except Exception as exc:
        logger.exception("Cortex REST error")
        # Fallback to SQL-based cortex for non-tool calls
        if not tools:
            try:
                from cortex_app.cortex import chat
                user_msg = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
                system_msg = next((m["content"] for m in messages if m.get("role") == "system"), None)
                history = [m for m in messages if m.get("role") in ("user", "assistant")][:-1]
                sql_result = chat(user_msg, system_prompt=system_msg, model=model, history=history)
                # Wrap in OpenAI format
                return JsonResponse({
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": sql_result.get("content", "")},
                        "finish_reason": "stop",
                    }],
                    "usage": sql_result.get("usage", {}),
                    "_dkk_backend": "snowflake-sql-fallback",
                })
            except Exception as fallback_exc:
                return JsonResponse({"error": {"message": str(fallback_exc), "type": "api_error"}}, status=500)

        return JsonResponse({"error": {"message": str(exc), "type": "api_error"}}, status=500)


@require_http_methods(["GET"])
def list_models(request):
    """GET /snowflake/v1/models"""
    return JsonResponse({
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": 1709000000, "owned_by": "snowflake-cortex"}
            for m in SUPPORTED_MODELS
        ]
    })


@csrf_exempt
@require_http_methods(["POST"])
def ask(request):
    """POST /snowflake/ask/ — simple {prompt: ...} shorthand"""
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    prompt = body.get("prompt") or body.get("message", "")
    if not prompt:
        return JsonResponse({"error": "'prompt' or 'message' required"}, status=400)

    model = body.get("model") or os.environ.get("CORTEX_MODEL", "claude-opus-4-6")

    try:
        from cortex_app.cortex import chat
        result = chat(prompt, model=model)
        return JsonResponse({"response": result.get("content", ""), "model": model})
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)


@require_http_methods(["GET"])
def health(request):
    return JsonResponse({
        "status": "ok",
        "service": "snowflake_app",
        "backend": "snowflake-cortex-rest",
        "default_model": os.environ.get("CORTEX_MODEL", "claude-opus-4-6"),
        "tool_calling": True,
    })
