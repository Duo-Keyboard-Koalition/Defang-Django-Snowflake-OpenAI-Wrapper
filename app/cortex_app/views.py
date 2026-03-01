"""
Cortex REST API views — Snowflake Cortex wrapper endpoints.

No OpenAI. Models run natively inside Snowflake Cortex.

Endpoints:
  POST /cortex/chat/        — single-turn or multi-turn chat
  GET  /cortex/models/      — list available Cortex models
  GET  /cortex/health/      — liveness check
"""

import json
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt

from .cortex import chat, list_models, DEFAULT_MODEL


@csrf_exempt
@require_http_methods(["POST"])
def cortex_chat(request):
    """
    Snowflake Cortex chat completion endpoint.

    Request body (JSON):
      {
        "message": "Your question here",           // required
        "system_prompt": "Optional override",      // optional
        "model": "snowflake-arctic",               // optional
        "history": [                               // optional
          {"role": "user",      "content": "..."},
          {"role": "assistant", "content": "..."}
        ]
      }

    Response:
      {
        "content": "...",
        "model": "snowflake-arctic",
        "usage": {...},
        "finish_reason": "stop"
      }
    """
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    user_message = body.get("message", "").strip()
    if not user_message:
        return JsonResponse({"error": "'message' field is required"}, status=400)

    system_prompt = body.get("system_prompt")
    model = body.get("model")
    history = body.get("history", [])

    if not isinstance(history, list):
        return JsonResponse({"error": "'history' must be a list"}, status=400)

    try:
        result = chat(
            user_message=user_message,
            system_prompt=system_prompt,
            model=model,
            history=history,
        )
        return JsonResponse(result)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)


@require_http_methods(["GET"])
def cortex_models(request):
    """List available Snowflake Cortex models."""
    models = list_models()
    return JsonResponse({
        "models": [{"id": m, "object": "model"} for m in models],
        "object": "list",
    })


@require_http_methods(["GET"])
def cortex_health(request):
    """Liveness / readiness check."""
    return JsonResponse({
        "status": "ok",
        "service": "cortex",
        "backend": "snowflake-cortex",
        "default_model": DEFAULT_MODEL,
    })
