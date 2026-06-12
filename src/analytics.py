import asyncio
import json
import logging
import re
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from jinja2 import Environment, StrictUndefined

from src.config import get_queue, get_settings
from src.llm import ModelClient
from src.session import Message, MessageRole, SessionClient
from src.telegram import TelegramMessage


logger = logging.getLogger(__name__)
settings = get_settings()
queue = get_queue()
session_client = SessionClient()
model_client = ModelClient()
jinja = Environment(
    autoescape=False,
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


@dataclass(frozen=True, slots=True)
class AnalyzerSpec:
    name: str
    instruction: str
    output_key: str
    output_schema: str
    min_user_messages: int = 1


FACTS_SPEC = AnalyzerSpec(
    name="user_fact_extraction",
    instruction=(
        "Extract only durable facts about the user that are worth remembering. "
        "Be conservative. Keep only high-confidence, stable facts. "
        "Do not guess. Do not duplicate already known facts. "
        "Useful categories include language, city, timezone, job, work, study, hobbies, routines, relationships, likes, dislikes, and similar durable facts."
    ),
    output_key="facts",
    output_schema=json.dumps(
        {
            "facts": [
                {
                    "key": "city",
                    "value": "Moscow",
                    "confidence": 0.94,
                    "source_ts": 1718000000,
                    "tags": ["travel"],
                }
            ]
        },
        ensure_ascii=False,
        indent=2,
    ),
    min_user_messages=2,
)

SUMMARY_SPEC = AnalyzerSpec(
    name="conversation_summary",
    instruction=(
        "Build a rolling summary of the recent chat context and relationship state. "
        "Focus on what was discussed recently, unresolved topics, promised follow-ups, emotional context, plans, and future callbacks. "
        "Keep it concise and practical."
    ),
    output_key="summary",
    output_schema=json.dumps(
        {
            "summary": "User recently discussed moving apartments, is stressed about work, and plans a weekend trip.",
            "open_loops": ["Ask how apartment search went", "Follow up on Saturday trip"],
            "relationship_context": "Supportive, ongoing friend chat.",
            "recent_threads": [
                {
                    "topic": "apartment search",
                    "status": "unfinished",
                    "tags": ["housing", "follow_up_needed"],
                }
            ],
        },
        ensure_ascii=False,
        indent=2,
    ),
    min_user_messages=2,
)

PREFERENCES_SPEC = AnalyzerSpec(
    name="user_preferences",
    instruction=(
        "Extract durable communication preferences and interaction style. "
        "Examples: preferred reply length, emoji tolerance, slang/formality level, whether they like jokes, flirting, bluntness, and which topics they engage with or shut down. "
        "Return only high-confidence preferences."
    ),
    output_key="preferences",
    output_schema=json.dumps(
        {
            "preferences": [
                {
                    "key": "reply_style",
                    "value": "short_casual",
                    "confidence": 0.88,
                    "source_ts": 1718000000,
                    "tags": ["style"],
                }
            ]
        },
        ensure_ascii=False,
        indent=2,
    ),
    min_user_messages=2,
)


def aclose() -> None:
    """
    Closes module-local clients.
    """
    session_client.close()
    model_client.close()


def _now_ts() -> int:
    return int(time.time())


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_tags(tags: Any) -> List[str]:
    if not isinstance(tags, list):
        return []

    result: List[str] = []

    for tag in tags:
        normalized = _normalize_text(tag).lower()

        if normalized and normalized not in result:
            result.append(normalized)

    return result


def _unique_by_text(items: Sequence[Dict[str, Any]], text_key: str = "text") -> List[Dict[str, Any]]:
    seen: set[str] = set()
    result: List[Dict[str, Any]] = []

    for item in items:
        text = _normalize_text(item.get(text_key))

        if not text or text in seen:
            continue

        seen.add(text)
        new_item = dict(item)
        new_item[text_key] = text
        result.append(new_item)

    return result


def _parse_json_response(text: str) -> Dict[str, Any]:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start >= 0 and end > start:
            parsed = json.loads(cleaned[start : end + 1])
        else:
            raise

    if not isinstance(parsed, dict):
        raise ValueError("Analytics model output must be a JSON object.")

    return parsed


def _serialize_messages(messages: Sequence[Message]) -> List[Dict[str, Any]]:
    return [
        {
            "role": message.role.value,
            "content": message.content,
            "timestamp": message.timestamp,
        }
        for message in messages
    ]


def _build_prompt(spec: AnalyzerSpec, context: Dict[str, Any]) -> str:
    template = model_client.load_prompt(settings.analytics_path)
    rendered_context = {
        "analyzer_name": spec.name,
        "analyzer_instruction": spec.instruction,
        "output_schema": spec.output_schema,
        "context_json": json.dumps(context, ensure_ascii=False, indent=2),
    }

    return jinja.from_string(template).render(rendered_context)


def _default_user_facts() -> Dict[str, Any]:
    return {"data": [], "updated_at": None}


def _default_user_preferences() -> Dict[str, Any]:
    return {"data": [], "updated_at": None}


def _default_conversation_summary() -> Dict[str, Any]:
    return {"summary": "", "open_loops": [], "relationship_context": "", "recent_threads": [], "updated_at": None}


def _default_behavior_snapshot() -> Dict[str, Any]:
    return {"updated_at": None}


def _default_conversation_memory() -> Dict[str, Any]:
    return {"likes": [], "events": [], "updated_at": None}


def _merge_keyed_entries(
    existing: Sequence[Dict[str, Any]],
    new_entries: Sequence[Dict[str, Any]],
    *,
    minimum_confidence: float,
) -> List[Dict[str, Any]]:
    merged: Dict[tuple[str, str], Dict[str, Any]] = {}

    for entry in existing:
        key = _normalize_text(entry.get("key")).casefold()
        value = _normalize_text(entry.get("value"))

        if not key or not value:
            continue

        merged[(key, value)] = {
            "key": _normalize_text(entry.get("key")),
            "value": value,
            "confidence": float(entry.get("confidence", 0.0) or 0.0),
            "source_ts": int(entry.get("source_ts", 0) or 0),
            "tags": _normalize_tags(entry.get("tags", [])),
        }

    for entry in new_entries:
        key = _normalize_text(entry.get("key")).casefold()
        value = _normalize_text(entry.get("value"))

        if not key or not value:
            continue

        confidence = float(entry.get("confidence", 0.0) or 0.0)

        if confidence < minimum_confidence:
            continue

        item = {
            "key": _normalize_text(entry.get("key")),
            "value": value,
            "confidence": confidence,
            "source_ts": int(entry.get("source_ts", _now_ts()) or _now_ts()),
            "tags": _normalize_tags(entry.get("tags", [])),
        }

        current = merged.get((key, value))

        if current is None or confidence >= float(current.get("confidence", 0.0) or 0.0):
            merged[(key, value)] = item

    result = list(merged.values())
    result.sort(key=lambda item: (int(item.get("source_ts", 0) or 0), _normalize_text(item.get("key"))))

    return result


def _merge_string_list(existing: Sequence[Any], new_items: Sequence[Any]) -> List[str]:
    merged: List[str] = []

    for item in list(existing) + list(new_items):
        text = _normalize_text(item)
        if text and text not in merged:
            merged.append(text)

    return merged


def _build_recent_messages_context(messages: Sequence[Message]) -> List[Dict[str, Any]]:
    return _serialize_messages(messages[-settings.analytics_history_limit :])


def _compute_sentiment_score(text: str) -> int:
    positive = {
        "good",
        "great",
        "nice",
        "love",
        "cool",
        "рад",
        "класс",
        "здорово",
        "норм",
        "awesome",
    }
    negative = {
        "bad",
        "sad",
        "hate",
        "annoy",
        "stress",
        "плохо",
        "бесит",
        "устал",
        "тяжело",
        "ужас",
    }

    words = {word.casefold() for word in re.findall(r"[\w\-]+", text, flags=re.UNICODE)}

    return sum(1 for word in words if word in positive) - sum(1 for word in words if word in negative)


def _contains_emoji(text: str) -> bool:
    return bool(re.search(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]|[:;]-?[)D(]|\)\)", text))


def _compute_behavior_snapshot(chat_id: int, history: Sequence[Message], now_ts: int) -> Dict[str, Any]:
    user_messages = [message for message in history if message.role == MessageRole.USER]
    texts = [_normalize_text(message.content) for message in user_messages if _normalize_text(message.content)]
    lengths = [len(text) for text in texts]
    avg_message_length = round(statistics.fmean(lengths), 2) if lengths else 0
    emoji_rate = round(sum(1 for text in texts if _contains_emoji(text)) / len(texts), 2) if texts else 0.0
    minutes_since_last_message = None

    if history:
        last_ts = int(history[-1].timestamp or 0)
        if last_ts > 0:
            minutes_since_last_message = max(0, int((now_ts - last_ts) / 60))

    previous_window = texts[: max(0, len(texts) - 5)]
    recent_window = texts[-5:]
    recent_score = sum(_compute_sentiment_score(text) for text in recent_window)
    previous_score = sum(_compute_sentiment_score(text) for text in previous_window[-5:]) if previous_window else 0

    if len(texts) < 4:
        sentiment_trend = "unknown"
        engagement_trend = "unknown"
        style_shift = False
        hints: List[str] = []
    else:
        sentiment_delta = recent_score - previous_score

        if sentiment_delta > 1:
            sentiment_trend = "more_positive"
        elif sentiment_delta < -1:
            sentiment_trend = "more_negative"
        else:
            sentiment_trend = "stable"

        recent_avg = statistics.fmean(len(text) for text in recent_window) if recent_window else avg_message_length
        previous_avg = statistics.fmean(len(text) for text in previous_window[-5:]) if previous_window else recent_avg

        if recent_avg < previous_avg * 0.7:
            engagement_trend = "lower"
        elif recent_avg > previous_avg * 1.3:
            engagement_trend = "higher"
        else:
            engagement_trend = "stable"

        style_shift = sentiment_trend != "stable" or engagement_trend != "stable" or abs(recent_avg - previous_avg) >= max(8.0, previous_avg * 0.3)
        hints = []

        if style_shift:
            hints.append("style_changed")

        if sentiment_trend in {"more_negative", "more_positive"}:
            hints.append(f"sentiment_{sentiment_trend}")

        if engagement_trend in {"lower", "higher"}:
            hints.append(f"engagement_{engagement_trend}")

    elapsed_seconds = 0

    if len(history) >= 2:
        first_ts = int(history[0].timestamp or 0)
        last_ts = int(history[-1].timestamp or 0)

        if first_ts > 0 and last_ts > first_ts:
            elapsed_seconds = last_ts - first_ts

    if elapsed_seconds > 0:
        message_frequency_per_hour = round(len(user_messages) * 3600 / elapsed_seconds, 2)
    else:
        message_frequency_per_hour = float(len(user_messages))

    summary = session_client.get_memory_document(chat_id, "conversation_summary", _default_conversation_summary())
    summary_updated_at = int(summary.get("updated_at") or 0)
    summary_stale_after = settings.analytics_summary_stale_after_minutes * 60
    summary_stale = bool(summary_updated_at and now_ts - summary_updated_at >= summary_stale_after)

    tags = []
    if summary_stale:
        tags.append("summary_stale")
    if sentiment_trend != "stable":
        tags.append("sentiment_shift")
    if engagement_trend != "stable":
        tags.append("engagement_shift")

    return {
        "avg_message_length": avg_message_length,
        "emoji_rate": emoji_rate,
        "message_frequency_per_hour": message_frequency_per_hour,
        "minutes_since_last_message": minutes_since_last_message,
        "summary_stale": summary_stale,
        "sentiment_trend": sentiment_trend,
        "engagement_trend": engagement_trend,
        "style_shift": style_shift,
        "hints": hints,
        "tags": tags,
        "updated_at": now_ts,
    }


def _derive_conversation_memory(
    chat_id: int,
    now_ts: int,
    facts: Dict[str, Any],
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    existing = session_client.get_memory_document(chat_id, "conversation_memory", _default_conversation_memory())
    existing_likes = existing.get("likes", []) if isinstance(existing, dict) else []
    existing_events = existing.get("events", []) if isinstance(existing, dict) else []

    likes: List[Dict[str, Any]] = []
    for fact in facts.get("data", []):
        key = _normalize_text(fact.get("key")).casefold()
        tags = set(_normalize_tags(fact.get("tags", [])))

        if key in {"likes", "like", "hobbies", "hobby", "interests"} or tags.intersection({"likes", "hobbies", "gaming", "travel", "food", "movies", "music", "sports"}):
            value = _normalize_text(fact.get("value"))
            if value:
                likes.append(
                    {
                        "text": f"User likes {value}.",
                        "source_ts": int(fact.get("source_ts", now_ts) or now_ts),
                        "confidence": float(fact.get("confidence", 0.0) or 0.0),
                        "tags": sorted(tags.union({"likes"})),
                    }
                )

    events: List[Dict[str, Any]] = []
    for loop in summary.get("open_loops", []):
        text = _normalize_text(loop)
        if text:
            events.append(
                {
                    "text": text,
                    "source_ts": int(summary.get("updated_at", now_ts) or now_ts),
                    "confidence": 0.75,
                    "tags": ["follow_up_needed"],
                }
            )

    for thread in summary.get("recent_threads", []):
        if not isinstance(thread, dict):
            continue
        topic = _normalize_text(thread.get("topic"))
        status = _normalize_text(thread.get("status"))
        if topic or status:
            text = "; ".join(part for part in [topic, status] if part)
            events.append(
                {
                    "text": text,
                    "source_ts": int(summary.get("updated_at", now_ts) or now_ts),
                    "confidence": 0.7,
                    "tags": _normalize_tags(thread.get("tags", [])) or ["conversation_thread"],
                }
            )

    merged_likes = _unique_by_text(list(existing_likes) + likes)
    merged_events = _unique_by_text(list(existing_events) + events)

    return {
        "likes": merged_likes,
        "events": merged_events,
        "updated_at": now_ts,
    }


async def _run_structured_analyzer(spec: AnalyzerSpec, chat_id: int, recent_history: Sequence[Message], bundle: Dict[str, Any], now_ts: int) -> Dict[str, Any]:
    context = {
        "chat_id": chat_id,
        "analysis_ts": now_ts,
        "recent_messages": _build_recent_messages_context(recent_history),
        "existing_memory": {
            "user_facts": bundle.get("user_facts", _default_user_facts()),
            "conversation_summary": bundle.get("conversation_summary", _default_conversation_summary()),
            "user_preferences": bundle.get("user_preferences", _default_user_preferences()),
            "behavior_snapshot": bundle.get("behavior_snapshot", _default_behavior_snapshot()),
            "conversation_memory": bundle.get("conversation_memory", _default_conversation_memory()),
        },
    }

    system_prompt = _build_prompt(spec, context)
    response = await model_client.chat(system_prompt, [Message(role=MessageRole.USER, content=json.dumps(context, ensure_ascii=False, indent=2))])
    return _parse_json_response(response)


async def _run_memory_analytics(chat_id: int, message_payloads: Sequence[Dict[str, Any]]) -> None:
    messages = [TelegramMessage.from_dict(payload) for payload in message_payloads if isinstance(payload, dict)]
    if not messages:
        logger.info("Skipping memory analytics for chat %s: no messages in payload", chat_id)
        return

    now_ts = _now_ts()
    history = session_client.get_history(chat_id)
    recent_history = history[-settings.analytics_history_limit :]
    user_messages = [message for message in recent_history if message.role == MessageRole.USER]

    if len(user_messages) < settings.analytics_min_user_messages:
        logger.info(
            "Skipping memory analytics for chat %s: only %s user messages available",
            chat_id,
            len(user_messages),
        )
        behavior_snapshot = _compute_behavior_snapshot(chat_id, recent_history, now_ts)
        session_client.set_memory_document(chat_id, "behavior_snapshot", behavior_snapshot)
        return

    bundle = session_client.get_memory_bundle(chat_id)

    behavior_snapshot = _compute_behavior_snapshot(chat_id, recent_history, now_ts)
    session_client.set_memory_document(chat_id, "behavior_snapshot", behavior_snapshot)

    if len(user_messages) >= FACTS_SPEC.min_user_messages:
        try:
            result = await _run_structured_analyzer(FACTS_SPEC, chat_id, recent_history, bundle, now_ts)
            facts = result.get("facts", [])

            if isinstance(facts, list):
                existing_facts = bundle.get("user_facts", {}).get("data", []) if isinstance(bundle.get("user_facts"), dict) else []
                merged = _merge_keyed_entries(existing_facts, facts, minimum_confidence=0.85)
                session_client.set_memory_document(chat_id, "user_facts", {"data": merged, "updated_at": now_ts})
                bundle["user_facts"] = {"data": merged, "updated_at": now_ts}
        except Exception:
            logger.exception("Memory fact extraction failed for chat %s", chat_id)

    if len(user_messages) >= SUMMARY_SPEC.min_user_messages:
        try:
            result = await _run_structured_analyzer(SUMMARY_SPEC, chat_id, recent_history, bundle, now_ts)
            summary_text = _normalize_text(result.get("summary"))
            open_loops = result.get("open_loops", [])
            recent_threads = result.get("recent_threads", [])
            relationship_context = _normalize_text(result.get("relationship_context"))

            existing_summary = bundle.get("conversation_summary", {}) if isinstance(bundle.get("conversation_summary"), dict) else {}
            existing_recent_threads = existing_summary.get("recent_threads", []) if isinstance(existing_summary.get("recent_threads"), list) else []
            merged_summary = {
                "summary": summary_text or _normalize_text(existing_summary.get("summary")),
                "open_loops": _merge_string_list(existing_summary.get("open_loops", []), open_loops) if isinstance(open_loops, list) else list(existing_summary.get("open_loops", [])),
                "relationship_context": relationship_context or _normalize_text(existing_summary.get("relationship_context")),
                "recent_threads": _unique_by_text(existing_recent_threads + (recent_threads if isinstance(recent_threads, list) else []), text_key="topic"),
                "updated_at": now_ts,
            }

            if not merged_summary["summary"]:
                merged_summary["summary"] = _normalize_text(existing_summary.get("summary"))

            session_client.set_memory_document(chat_id, "conversation_summary", merged_summary)
            bundle["conversation_summary"] = merged_summary
        except Exception:
            logger.exception("Conversation summary extraction failed for chat %s", chat_id)

    if len(user_messages) >= PREFERENCES_SPEC.min_user_messages:
        try:
            result = await _run_structured_analyzer(PREFERENCES_SPEC, chat_id, recent_history, bundle, now_ts)
            preferences = result.get("preferences", [])

            if isinstance(preferences, list):
                existing_preferences = bundle.get("user_preferences", {}).get("data", []) if isinstance(bundle.get("user_preferences"), dict) else []
                merged = _merge_keyed_entries(existing_preferences, preferences, minimum_confidence=0.75)
                session_client.set_memory_document(chat_id, "user_preferences", {"data": merged, "updated_at": now_ts})
                bundle["user_preferences"] = {"data": merged, "updated_at": now_ts}
        except Exception:
            logger.exception("User preference extraction failed for chat %s", chat_id)

    conversation_memory = _derive_conversation_memory(
        chat_id,
        now_ts,
        bundle.get("user_facts", _default_user_facts()),
        bundle.get("conversation_summary", _default_conversation_summary()),
    )
    session_client.set_memory_document(chat_id, "conversation_memory", conversation_memory)


def run_memory_analytics(chat_id: int, message_payloads: Sequence[Dict[str, Any]]) -> None:
    """
    RQ job entrypoint for memory analytics.
    """
    asyncio.run(_run_memory_analytics(chat_id, message_payloads))


def _should_trigger_by_message_count(chat_id: int) -> bool:
    count = session_client.get_memory_counter(chat_id, "user_message_count", 0)
    interval = max(1, settings.analytics_user_message_interval)
    return count > 0 and count % interval == 0


def _should_trigger_by_minute(chat_id: int, message_payloads: Sequence[Dict[str, Any]]) -> bool:
    interval = max(1, settings.analytics_minute_interval)
    message_dates = [int(payload.get("date") or 0) for payload in message_payloads if isinstance(payload, dict)]

    if not message_dates:
        return False

    bucket = max(message_dates) // 60
    if bucket <= 0 or bucket % interval != 0:
        return False

    last_bucket = session_client.get_memory_counter(chat_id, "last_minute_bucket", 0)
    if last_bucket == bucket:
        return False

    session_client.set_memory_document(chat_id, "last_minute_bucket", bucket)
    return True


def enqueue_memory_analytics(chat_id: int, messages: Sequence[TelegramMessage]) -> None:
    """
    Enqueues analytics work for a chat when the configured triggers fire.
    """
    payloads = [message.to_dict() for message in messages]
    if not payloads:
        return

    user_message_count = sum(1 for message in messages if message.chat_id == chat_id and (message.text or "").strip())
    if user_message_count:
        session_client.increment_memory_counter(chat_id, "user_message_count", user_message_count)

    should_run = False
    reasons: List[str] = []

    if _should_trigger_by_message_count(chat_id):
        should_run = True
        reasons.append(f"messages_{session_client.get_memory_counter(chat_id, 'user_message_count', 0)}")

    if _should_trigger_by_minute(chat_id, payloads):
        should_run = True
        reasons.append(f"minute_{max(int(payload.get('date') or 0) for payload in payloads if isinstance(payload, dict)) // 60}")

    if not should_run:
        return

    job_id = "memory_analytics_{}".format("_".join([str(chat_id)] + reasons))
    queue.enqueue(run_memory_analytics, chat_id, payloads, job_id=job_id)
