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
    system_prompt_template_path: str = "./llm/system.txt"
    llm_params_path: str = "./llm/params.json"
    persona_catalog_path: str = "./llm/personas.json"
    persona_template_path: str = "./llm/persona.txt"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # History
    max_history_messages: int = 12

    # Environment
    environment: str = "development"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
