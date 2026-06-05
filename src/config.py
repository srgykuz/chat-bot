from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from .env file and environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False
    )

    telegram_token: str = Field(
        default="",
        description="Telegram bot token from BotFather.",
    )
    telegram_use_polling: bool = Field(
        default=False,
        description="Use long polling to receive updates instead of expecting webhook endpoint call.",
    )

    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API key.",
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key.",
    )

    system_path: str = Field(
        default="./system",
        description="Path to the directory that stores system prompt and model params.",
    )
    personas_path: str = Field(
        default="./personas",
        description="Path to the directory that stores persona definitions.",
    )

    redis_url: str = Field(
        default="redis://redis:6379",
        description="Redis connection URL.",
    )

    history_limit: int = Field(
        default=50,
        description="Maximum number of recent messages to keep in chat history per user.",
    )


@lru_cache()
def get_settings() -> Settings:
    """Returns parsed and validated settings instance."""

    return Settings()


if __name__ == "__main__":
    settings = get_settings()

    print(settings.model_dump_json(indent=4))
