from django.apps import AppConfig


class SnowflakeAppConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "snowflake_app"
    verbose_name = "Snowflake Cortex Wrapper"
