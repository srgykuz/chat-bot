from datetime import datetime, timezone, timedelta

from src.config import get_settings, get_redis


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


def minute() -> str:
    """
    Returns current time in UTC timezone in ISO format (HH:MM).
    """
    return now().strftime("%H:%M")


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


def should_limit_chat(chat_id: int) -> bool:
    """
    Checks all related chat metrics and decides if processing messages in this chat
    should be paused until this function returns `False`.

    If all limits values are equal to zero, then it means that limiting is disabled
    and this function always will return `False`.

    Should be called before LLM API request.
    """
    settings = get_settings()
    rpm_limit = settings.limit_chat_rpm
    rpd_limit = settings.limit_chat_rpd
    tpm_limit = settings.limit_chat_tpm
    tpd_limit = settings.limit_chat_tpd

    if (rpm_limit + rpd_limit + tpm_limit + tpd_limit) == 0:
        return False

    rpm_key = key_chat_rpm(chat_id)
    rpd_key = key_chat_rpd(chat_id)
    tpm_key = key_chat_tpm(chat_id)
    tpd_key = key_chat_tpd(chat_id)
    pipe = get_redis().pipeline()

    pipe.get(rpm_key)
    pipe.get(rpd_key)
    pipe.get(tpm_key)
    pipe.get(tpd_key)

    rpm_val, rpd_val, tpm_val, tpd_val = pipe.execute()

    if (rpm_limit > 0) and (rpm_val is not None) and (int(rpm_val) >= rpm_limit):
        return True

    if (rpd_limit > 0) and (rpd_val is not None) and (int(rpd_val) >= rpd_limit):
        return True

    if (tpm_limit > 0) and (tpm_val is not None) and (int(tpm_val) >= tpm_limit):
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


def track_chat_rpd(chat_id: int):
    """
    Tracks chat requests per day metric.

    Should be called after LLM API request.
    """
    key = key_chat_rpd(chat_id)
    pipe = get_redis().pipeline()

    pipe.incr(key)
    pipe.expire(key, timedelta(days=2), nx=True)

    pipe.execute()


def key_chat_rpd(chat_id: int) -> str:
    """
    Composes a key for `track_chat_rpd()`.
    """
    return f"limiter:chat:{chat_id}:rpd:{date()}"


def track_chat_tpd(chat_id: int, tokens: int):
    """
    Tracks chat total tokens usage (input + output) per day metric.

    Should be called after LLM API request.
    """
    key = key_chat_tpd(chat_id)
    pipe = get_redis().pipeline()

    pipe.incrby(key, tokens)
    pipe.expire(key, timedelta(days=2), nx=True)

    pipe.execute()


def key_chat_tpd(chat_id: int) -> str:
    """
    Composes a key for `track_chat_tpd()`.
    """
    return f"limiter:chat:{chat_id}:tpd:{date()}"


def track_chat_rpm(chat_id: int):
    """
    Tracks chat requests per minute metric.

    Should be called after LLM API request.
    """
    key = key_chat_rpm(chat_id)
    pipe = get_redis().pipeline()

    pipe.incr(key)
    pipe.expire(key, timedelta(minutes=2), nx=True)

    pipe.execute()


def key_chat_rpm(chat_id: int) -> str:
    """
    Composes a key for `track_chat_rpm()`.
    """
    return f"limiter:chat:{chat_id}:rpm:{minute()}"


def track_chat_tpm(chat_id: int, tokens: int):
    """
    Tracks chat total tokens usage (input + output) per minute metric.

    Should be called after LLM API request.
    """
    key = key_chat_tpm(chat_id)
    pipe = get_redis().pipeline()

    pipe.incrby(key, tokens)
    pipe.expire(key, timedelta(minutes=2), nx=True)

    pipe.execute()


def key_chat_tpm(chat_id: int) -> str:
    """
    Composes a key for `track_chat_tpm()`.
    """
    return f"limiter:chat:{chat_id}:tpm:{minute()}"
