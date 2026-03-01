from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json
import os

CORTEX_MODELS = [
    {"id": "claude-opus-4-6",   "label": "Claude Opus 4.6 (Cortex)"},
    {"id": "claude-opus-4-5",   "label": "Claude Opus 4.5 (Cortex)"},
    {"id": "snowflake-arctic",  "label": "Snowflake Arctic"},
    {"id": "mistral-large2",    "label": "Mistral Large 2"},
    {"id": "llama3.1-70b",      "label": "Llama 3.1 70B"},
    {"id": "llama3.1-8b",       "label": "Llama 3.1 8B"},
    {"id": "gemma-7b",          "label": "Gemma 7B"},
    {"id": "mixtral-8x7b",      "label": "Mixtral 8x7B"},
]


def chat_ui(request):
    """Render the main chat window."""
    return render(request, 'frontend_app/chat.html', {
        'models': CORTEX_MODELS,
        'default_model': os.getenv('CORTEX_MODEL', 'claude-opus-4-6'),
    })


@csrf_exempt
@require_http_methods(["POST"])
def chat_proxy(request):
    """
    Calls Snowflake Cortex directly (no HTTP round-trip).
    """
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    message = body.get("message", "").strip()
    if not message:
        return JsonResponse({"error": "'message' is required"}, status=400)

    model = body.get("model")
    history = body.get("history", [])
    system_prompt = body.get("system_prompt")

    try:
        from cortex_app.cortex import chat
        result = chat(message, system_prompt=system_prompt, model=model, history=history)
        return JsonResponse(result)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)


@require_http_methods(["GET"])
def models_list(request):
    """Return available models for the frontend dropdown."""
    return JsonResponse({"models": CORTEX_MODELS})
