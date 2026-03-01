"""
DKK Chat Endpoint — Django view
Proxies to GalClaw's RAG /ask agent on 192.168.16.113:5000
Persists history to Postgres
Also exposes OpenAI-compatible /v1/chat/completions
"""

import uuid
import time
import json
import requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .models import ChatSession, ChatMessage

RAG_ENDPOINT = "http://192.168.16.113:5000/v1/chat/completions"
RAG_TIMEOUT = 30


def _call_rag(question: str, caller: str) -> dict:
    resp = requests.post(
        RAG_ENDPOINT,
        json={
            "model": "mistral-large",
            "messages": [{"role": "user", "content": question}],
            "user": caller,
        },
        timeout=RAG_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "answer": data["choices"][0]["message"]["content"],
        "sources": data.get("sources", []),
    }


@csrf_exempt
@require_http_methods(["POST"])
def chat(request):
    """
    Simple chat endpoint.
    POST /chat/ask/
    Body: {"message": "...", "caller": "bot-name", "session_id": "optional"}
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "invalid JSON"}, status=400)

    message = body.get("message") or body.get("question") or body.get("query")
    if not message:
        return JsonResponse({"error": "message/question/query field required"}, status=400)

    caller = body.get("caller", "unknown")
    session_id = body.get("session_id") or str(uuid.uuid4())

    session, _ = ChatSession.objects.get_or_create(
        session_id=session_id, defaults={"caller": caller}
    )

    # Save user message
    ChatMessage.objects.create(session=session, role="user", content=message)

    # Hit RAG agent
    try:
        rag_result = _call_rag(message, caller)
        answer = rag_result.get("answer", "No answer returned.")
        sources = rag_result.get("sources", [])
    except Exception as e:
        answer = f"RAG agent unavailable: {e}"
        sources = []

    # Save assistant response
    ChatMessage.objects.create(
        session=session, role="assistant", content=answer, sources=sources
    )

    return JsonResponse({
        "answer": answer,
        "sources": sources,
        "session_id": session_id,
        "caller": caller,
    })


@csrf_exempt
@require_http_methods(["POST"])
def openai_chat_completions(request):
    """
    OpenAI-compatible endpoint.
    POST /chat/v1/chat/completions
    Body: standard OpenAI messages format
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": {"message": "invalid JSON", "type": "invalid_request_error"}}, status=400)

    messages = body.get("messages", [])
    if not messages:
        return JsonResponse({"error": {"message": "messages field required", "type": "invalid_request_error"}}, status=400)

    # Extract last user message
    user_message = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"), None
    )
    if not user_message:
        return JsonResponse({"error": {"message": "no user message found", "type": "invalid_request_error"}}, status=400)

    caller = body.get("user", "openai-client")
    model = body.get("model", "cortex-rag")

    try:
        rag_result = _call_rag(user_message, caller)
        answer = rag_result.get("answer", "No answer returned.")
        sources = rag_result.get("sources", [])
    except Exception as e:
        answer = f"RAG agent unavailable: {e}"
        sources = []

    # OpenAI-format response
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    return JsonResponse({
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": answer,
            },
            "logprobs": None,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(user_message.split()),
            "completion_tokens": len(answer.split()),
            "total_tokens": len(user_message.split()) + len(answer.split()),
        },
        "system_fingerprint": None,
        # DKK extension — sources for context
        "_dkk_sources": sources,
    })


@require_http_methods(["GET"])
def list_models(request):
    """GET /chat/v1/models — OpenAI-compatible model list"""
    return JsonResponse({
        "object": "list",
        "data": [
            {"id": "cortex-rag", "object": "model", "created": 1709000000, "owned_by": "dkk"},
            {"id": "mistral-large2", "object": "model", "created": 1709000000, "owned_by": "snowflake-cortex"},
            {"id": "llama3.1-70b", "object": "model", "created": 1709000000, "owned_by": "snowflake-cortex"},
        ]
    })


@require_http_methods(["GET"])
def health(request):
    return JsonResponse({"status": "ok", "rag_endpoint": RAG_ENDPOINT})


@require_http_methods(["GET"])
def history(request, session_id):
    """GET /chat/history/<session_id>/ — fetch chat history"""
    try:
        session = ChatSession.objects.get(session_id=session_id)
    except ChatSession.DoesNotExist:
        return JsonResponse({"error": "session not found"}, status=404)

    messages = [
        {"role": m.role, "content": m.content, "sources": m.sources, "created_at": m.created_at.isoformat()}
        for m in session.messages.all()
    ]
    return JsonResponse({"session_id": session_id, "caller": session.caller, "messages": messages})
