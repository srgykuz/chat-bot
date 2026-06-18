import json
import logging
from dataclasses import dataclass, asdict
from typing import cast, Any

from src.config import get_settings, get_redis, get_httpx


logger = logging.getLogger(__name__)
redis = get_redis()
httpx = get_httpx()


@dataclass(frozen=True, slots=True)
class WeatherInfo:
    """
    Information about current weather in a city.

    See https://www.weatherapi.com/docs/
    """
    city: str
    temp_c: float
    condition_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WeatherInfo":
        city = str(data.get("city", ""))
        temp_c = float(data.get("temp_c", 0.0))
        condition_text = str(data.get("condition_text", ""))

        return cls(
            city=city,
            temp_c=temp_c,
            condition_text=condition_text,
        )


async def fetch_weather(city: str, lang: str = "", use_cache: bool = True) -> WeatherInfo:
    """
    Fetches current weather information for a given city using API.

    lang can be a language code (e.g. "de"), empty string means English.
    If use_cache is True, the function will cache a result.
    """
    city = (city or "").strip()

    if not city:
        raise ValueError("City name is required")

    settings = get_settings()

    if not settings.weatherapi_api_key:
        raise RuntimeError("WeatherAPI API key is not configured")

    cache_key = f"weather:{city.casefold()}"

    if use_cache:
        cached = cast(str | None, redis.get(cache_key))

        if cached:
            data = json.loads(cached)
            info = WeatherInfo.from_dict(data)

            return info

    params = {
        "key": settings.weatherapi_api_key,
        "q": city,
        "aqi": "no",
    }

    if lang:
        params["lang"] = lang

    response = await httpx.get(
        "https://api.weatherapi.com/v1/current.json",
        params=params,
    )

    response.raise_for_status()

    payload = response.json()
    error = payload.get("error")

    if error:
        raise RuntimeError(f"Error fetching weather info for city={city}: {error}")

    current = payload.get("current", {})
    temp_c = current.get("temp_c", 0.0)
    condition = current.get("condition", {})
    condition_text = condition.get("text", "")

    info = WeatherInfo(
        city=city,
        temp_c=temp_c,
        condition_text=condition_text,
    )

    if use_cache:
        data = json.dumps(info.to_dict(), ensure_ascii=False)

        redis.setex(
            cache_key,
            settings.weatherapi_cache_ttl,
            data,
        )

    return info


if __name__ == "__main__":
    import asyncio

    city = "Tokyo"
    info = asyncio.run(fetch_weather(city, use_cache=False))
    data = info.to_dict()

    print(data)
