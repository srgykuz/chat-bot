"""Configuration management for the friend bot."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Telegram
    telegram_token: str = ""
    telegram_webhook_url: str = ""
    telegram_use_polling: bool = False

    # LLM APIs
    gemini_api_key: str = ""
    openai_api_key: str = ""

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Environment
    environment: str = "development"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
