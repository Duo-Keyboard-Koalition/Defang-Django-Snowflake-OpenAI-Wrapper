from django.urls import path
from . import views

app_name = "snowflake_app"

urlpatterns = [
    path("v1/chat/completions", views.chat_completions, name="chat_completions"),
    path("v1/models", views.list_models, name="list_models"),
    path("ask/", views.ask, name="ask"),
    path("health/", views.health, name="health"),
]
