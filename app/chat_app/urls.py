from django.urls import path
from . import views

urlpatterns = [
    path("ask/", views.chat, name="chat-ask"),
    path("health/", views.health, name="chat-health"),
    path("history/<str:session_id>/", views.history, name="chat-history"),
    # OpenAI-compatible routes
    path("v1/chat/completions", views.openai_chat_completions, name="openai-completions"),
    path("v1/models", views.list_models, name="openai-models"),
]
