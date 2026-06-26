import asyncio
import logging
from datetime import timedelta
from time import time
import statistics

from src.config import get_settings
from src.session import SessionClient
from src.llm import ModelClient
from src.schema import Message, EmotionalState, EmotionalStateLLM, Facts, ConversationSummary, ConversationSummaryLLM


logger = logging.getLogger(__name__)
settings = get_settings()
session_client = SessionClient()
model_client = ModelClient("analytics")

analyze_chat_1m_timedelta = timedelta(minutes=1)
analyze_chat_3m_timedelta = timedelta(minutes=3)
analyze_chat_5m_timedelta = timedelta(minutes=5)


def close() -> None:
    """
    Closes underlying resources.
    """
    session_client.close()
    model_client.close()


def analyze_chat_1m(chat_id: int) -> None:
    """
    Executes set of chat analyzers.

    This function intended to be executed in 1 minute since a user's message.
    Only one instance of this function should be scheduled at a moment.
    """
    history = session_client.get_history(chat_id)

    if len(history) < 10:
        logger.info("Skipping due to short history")
        return

    infer_emotional_state(chat_id, history)


def analyze_chat_3m(chat_id: int) -> None:
    """
    Executes set of chat analyzers.

    This function intended to be executed in 3 minutes since a user's message.
    Only one instance of this function should be scheduled at a moment.
    """
    history = session_client.get_history(chat_id)

    if len(history) < 10:
        logger.info("Skipping due to short history")
        return

    infer_facts(chat_id, history)


def analyze_chat_5m(chat_id: int) -> None:
    """
    Executes set of chat analyzers.

    This function intended to be executed in 5 minutes since a user's message.
    Only one instance of this function should be scheduled at a moment.
    """
    history = session_client.get_history(chat_id)

    if len(history) < 10:
        logger.info("Skipping due to short history")
        return

    infer_conversation_summary(chat_id, history)


def history_to_conversation(history: list[Message]) -> str:
    """
    Converts chat history into format suitable for LLM.
    """
    return "\n".join([f"{msg.role.value.title()}: {msg.content}" for msg in history])


def infer_emotional_state(chat_id: int, history: list[Message]) -> None:
    """
    Infers user's emotional state and updates the storage.
    """
    system_prompt = (
        "You are a backend analysis engine. Your task is to review the provided "
        "conversation history between user and assistant and infer user's current "
        "emotional state. Use all provided messages as evidence and conclude the "
        "most likely mood, tone, and engagement of the user. Use assistant messages "
        "only as context. Output a JSON object that strictly matches the requested "
        "schema. Set confidence fields to reflect how certain you are about each value."
    )
    user_prompt = history_to_conversation(history[-10:])

    result = asyncio.run(model_client.generate(
        system_prompt,
        user_prompt,
        response_format=EmotionalStateLLM,
    ))
    state = EmotionalState.loads(result.content)

    session_client.append_emotional_states(chat_id, state, 5)

    states = session_client.get_emotional_states(chat_id)
    now = time()
    recency = timedelta(hours=1)
    recent_states = [s for s in states if (now - s.timestamp) <= recency.total_seconds()]

    if len(recent_states) < 3:
        return

    mood = statistics.mode([s.mood for s in recent_states])
    tone = statistics.mode([s.tone for s in recent_states])
    engagement = statistics.mode([s.engagement for s in recent_states])
    mood_confidence = statistics.mean([s.mood_confidence for s in recent_states if s.mood == mood])
    tone_confidence = statistics.mean([s.tone_confidence for s in recent_states if s.tone == tone])
    engagement_confidence = statistics.mean([s.engagement_confidence for s in recent_states if s.engagement == engagement])

    current_state = EmotionalState(
        mood=mood,
        tone=tone,
        engagement=engagement,
        mood_confidence=mood_confidence,
        tone_confidence=tone_confidence,
        engagement_confidence=engagement_confidence,
    )
    expires = timedelta(hours=1)

    session_client.set_emotional_state(chat_id, current_state, expires)


def infer_facts(chat_id: int, history: list[Message]) -> None:
    """
    Infers facts about a user and updates the storage.
    """
    known_facts = session_client.get_facts(chat_id)
    known_facts_s = known_facts.dumps() if known_facts else ""

    system_prompt = (
        "You are a backend analysis engine. Your task is to review the provided "
        "conversation history between user and assistant and extract only new factual "
        "information about the user. You will receive a list of already known facts. "
        "Do not repeat facts that are already known, even if they are phrased differently. "
        "Treat facts as duplicates if they refer to the same underlying meaning, even with "
        "different wording, spelling, language, or granularity. If a fact is partially "
        "overlapping with an existing fact, return it only if it adds materially new "
        "information. Output values in the user's language. Use assistant messages only for "
        "context. Output your evaluation strictly matching the requested JSON schema."
        "\n\n"
        f"Known facts: {known_facts_s}"
    )
    user_prompt = history_to_conversation(history)

    result = asyncio.run(model_client.generate(
        system_prompt,
        user_prompt,
        response_format=Facts,
    ))
    new_facts = Facts.loads(result.content)

    if known_facts:
        new_facts.facts.extend(known_facts.facts)

    new_facts.facts = new_facts.facts[:settings.facts_limit]

    session_client.set_facts(chat_id, new_facts)


def infer_conversation_summary(chat_id: int, history: list[Message]) -> None:
    """
    Summarizes the conversation into multiple summaries and updates the storage.
    """
    known_summary = session_client.get_conversation_summary(chat_id)
    known_summary_llm = known_summary.to_llm() if known_summary else None
    known_summary_s = known_summary_llm.dumps() if known_summary_llm else ""

    system_prompt = (
        "You are a backend analysis engine. Your task is to review the provided "
        "conversation history between user and assistant and summarize the conversation "
        "into one or more summaries that are worth to remember or that add new information "
        "to already known summaries. One topic per one summary. You will receive a list of "
        "already known summaries. Do not repeat summaries that are already known, even if "
        "they are phrased differently. Treat summaries as duplicates if they refer to the "
        "same underlying meaning, even with different wording, spelling, language, or granularity. "
        "If a new summary is partially overlapping with an existing summary, return it only if it adds "
        "materially new information. Output values in the user's language. Use assistant messages "
        "only for context. Output your evaluation strictly matching the requested JSON schema. "
        "Return an empty result if there is nothing useful to remember or if the conversation does not "
        "have at least one clear topic."
        "\n\n"
        f"Known summaries: {known_summary_s}"
    )
    user_prompt = history_to_conversation(history)

    result = asyncio.run(model_client.generate(
        system_prompt,
        user_prompt,
        response_format=ConversationSummaryLLM,
    ))

    now = time()
    new_summary_llm = ConversationSummaryLLM.loads(result.content)
    new_summary = ConversationSummary(
        summaries=new_summary_llm.summaries,
        timestamps=[now for _ in new_summary_llm.summaries]
    )

    if known_summary:
        new_summary.summaries.extend(known_summary.summaries)
        new_summary.timestamps.extend(known_summary.timestamps)

    new_summary.summaries = new_summary.summaries[:settings.summaries_limit]
    new_summary.timestamps = new_summary.timestamps[:settings.summaries_limit]

    session_client.set_conversation_summary(chat_id, new_summary)
