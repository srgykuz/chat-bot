from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from redis import Redis
from rq import Queue
import httpx


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

    google_api_key: str = Field(
        default="",
        description="Google API key.",
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key.",
    )
    ollama_host: str = Field(
        default="",
        description="Ollama host URL (e.g. http://localhost:11434 or https://ollama.com).",
    )
    ollama_api_key: str = Field(
        default="",
        description="Ollama API key.",
    )

    weatherapi_api_key: str = Field(
        default="",
        description="https://www.weatherapi.com API key.",
    )
    weatherapi_cache_ttl: int = Field(
        default=15 * 60,
        description="Time in seconds to cache fetched weather info.",
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
    facts_limit: int = Field(
        default=50,
        description="Maximum number of recent facts to keep in chat history per user."
    )
    chat_flush_interval: int = Field(
        default=5,
        description="Time in seconds to wait for additional user messages before flushing the buffered batch.",
    )
    output_separator: str = Field(
        default="[SPLIT]",
        description="Separator string to split LLM response into multiple messages.",
    )


@lru_cache()
def get_settings() -> Settings:
    """Returns parsed and validated settings instance."""

    return Settings()


@lru_cache()
def get_redis(decode_responses: bool = True) -> Redis:
    """Returns a Redis client instance based on the settings."""
    settings = get_settings()

    return Redis.from_url(settings.redis_url, decode_responses=decode_responses)


@lru_cache()
def get_queue() -> Queue:
    """Returns an RQ Queue instance based on the settings."""
    redis = get_redis(decode_responses=False)

    return Queue("default", connection=redis)


@lru_cache()
def get_httpx() -> httpx.AsyncClient:
    """Returns an HTTPX AsyncClient instance."""

    return httpx.AsyncClient(timeout=10)


if __name__ == "__main__":
    settings = get_settings()

    print(settings.model_dump_json(indent=4))
