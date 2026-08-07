from datetime import datetime, timezone, timedelta

from src.config import get_redis


def now() -> datetime:
    """
    Returns current time in UTC timezone.
    """
    return datetime.now(timezone.utc)


def date() -> str:
    """
    Returns current date in UTC timezone in ISO format (YYYY-MM-DD).
    """
    return now().date().isoformat()


def should_limit_llm(name: str, rpd_limit: int, tpd_limit: int) -> bool:
    """
    Checks all related metrics and decides if LLM API usage on behalf of client
    with given name should be paused until this function returns `False`.

    If all limits values are equal to zero, then it means that limiting is disabled and
    this function always will return `False`.

    Should be called before LLM API request.
    """
    if (rpd_limit + tpd_limit) == 0:
        return False

    rpd_key = key_llm_rpd(name)
    tpd_key = key_llm_tpd(name)
    pipe = get_redis().pipeline()

    pipe.get(rpd_key)
    pipe.get(tpd_key)

    rpd_val, tpd_val = pipe.execute()

    if (rpd_limit > 0) and (rpd_val is not None) and (int(rpd_val) >= rpd_limit):
        return True

    if (tpd_limit > 0) and (tpd_val is not None) and (int(tpd_val) >= tpd_limit):
        return True

    return False


def track_llm_rpd(name: str):
    """
    Tracks LLM requests per day metric.

    Should be called after LLM API request.
    """
    key = key_llm_rpd(name)
    pipe = get_redis().pipeline()

    pipe.incr(key)
    pipe.expire(key, timedelta(days=2), nx=True)

    pipe.execute()


def key_llm_rpd(name: str) -> str:
    """
    Composes a key for `track_llm_rpd()`.
    """
    return f"limiter:llm:{name}:rpd:{date()}"


def track_llm_tpd(name: str, tokens: int):
    """
    Tracks LLM total tokens usage (input + output) per day metric.

    Should be called after LLM API request.
    """
    key = key_llm_tpd(name)
    pipe = get_redis().pipeline()

    pipe.incrby(key, tokens)
    pipe.expire(key, timedelta(days=2), nx=True)

    pipe.execute()


def key_llm_tpd(name: str) -> str:
    """
    Composes a key for `track_llm_tpd()`.
    """
    return f"limiter:llm:{name}:tpd:{date()}"
