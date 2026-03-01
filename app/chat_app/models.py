from django.db import models


class ChatSession(models.Model):
    session_id = models.CharField(max_length=128, unique=True)
    caller = models.CharField(max_length=64, default="unknown")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.caller}:{self.session_id}"


class ChatMessage(models.Model):
    ROLE_CHOICES = [("user", "user"), ("assistant", "assistant")]

    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=16, choices=ROLE_CHOICES)
    content = models.TextField()
    sources = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"[{self.role}] {self.content[:60]}"
