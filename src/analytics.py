import asyncio
from datetime import timedelta
from time import time
import statistics

from src.config import get_logger, get_settings
from src.session import SessionClient
from src.llm import ModelClient, history_to_conversation
from src.schema import (
    Message,
    EmotionalState,
    EmotionalStateLLM,
    Facts,
    ConversationSummary,
    ConversationSummaryLLM,
    Relationships,
    RelationshipsDelta,
    Flag,
)


logger = get_logger(__name__)
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
    infer_relationships(chat_id, history)


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


def infer_relationships(chat_id: int, history: list[Message]) -> None:
    """
    Changes current relationships based on the conversation tone and updates the storage.
    """
    system_prompt = (
        "You are a backend analysis engine. Your task is to review the provided "
        "conversation history between user and assistant and infer how the assistant's "
        "relationship with the user should change as a result of this conversation. "
        "Evaluate all dimensions independently: friendship, trust, and romance. "
        "Use positive deltas when the conversation clearly strengthens that dimension, "
        "negative deltas when it clearly weakens that dimension, and 0 when there is "
        "not enough evidence for change. Use assistant messages only as context. Base your "
        "judgment primarily on the user's reactions, tone, cooperation, warmth, openness, "
        "appreciation, hostility, comfort, and boundaries shown in the conversation. "
        "Do not infer romance increase from ordinary politeness or friendliness alone. "
        "Do not punish brief misunderstandings unless they noticeably damage the interaction. "
        "Keep changes conservative: use small adjustments for mild signals and stronger "
        "adjustments only for clear repeated evidence. Return 0 for all fields if the "
        "conversation is neutral, ambiguous, or too short to justify a meaningful update. "
        "Consider the overall interaction trend, not any single line in isolation. "
        "Output a JSON object that strictly matches the requested schema."
    )
    user_prompt = history_to_conversation(history[-15:])

    result = asyncio.run(model_client.generate(
        system_prompt,
        user_prompt,
        response_format=RelationshipsDelta,
    ))
    delta = RelationshipsDelta.loads(result.content)

    current = session_client.get_relationships(chat_id)

    if not current:
        current = Relationships(
            friendship=0,
            trust=0,
            romance=0,
        )

    updated = Relationships(
        friendship=max(-100, min(100, current.friendship + delta.friendship_delta)),
        trust=max(-100, min(100, current.trust + delta.trust_delta)),
        romance=max(-100, min(100, current.romance + delta.romance_delta)),
    )

    session_client.set_relationships(chat_id, updated)


async def is_illegal_assistant(history: list[Message]) -> bool:
    """
    Analyzes history to find if any of the assistant messages contain or imply illegal content.

    Only assistant messages are subject for search. User messages are used only for context and
    not cause True result even if they contain illegal content.

    It is preferable to pass concrete few new messages that should be analyzed
    instead of passing full history that most of which was already analyzed.
    """
    system_prompt = (
        "You are a backend safety moderation engine. Your task is to review the provided "
        "conversation history between user and assistant and determine whether any assistant "
        "message contains, promotes, instructs, facilitates, normalizes, or meaningfully "
        "encourages illegal, highly dangerous, or prohibited content. User messages are "
        "provided only as context. The primary target of this review is assistant responses."
        "\n\n"
        "Set flag to True if at least one assistant message does any of the following:\n"
        "- provides instructions, steps, recipes, tactics, operational advice, procurement advice, or troubleshooting;\n"
        "- recommends, endorses, justifies, or encourages the act;\n"
        "- gives actionable assistance that would make the act easier to carry out;\n"
        "- meaningfully helps evade law enforcement, detection, attribution, or safety controls."
        "\n\n"
        "The content categories to detect include:\n"
        "- pornography or sexual exploitation, especially anything involving minors or coercion;\n"
        "- violence, violent wrongdoing, murder, torture, kidnapping, or assault;\n"
        "- illegal drugs, narcotics production, trafficking, distribution, or concealment;\n"
        "- weapons creation, acquisition, modification, trafficking, or use for harm;\n"
        "- suicide, self-harm, or instructions that facilitate death or severe injury;\n"
        "- fraud, scams, identity theft, forgery, money laundering, or financial crime;\n"
        "- hacking, malware, credential theft, phishing, unauthorized access, or cyber abuse;\n"
        "- stalking, doxxing, harassment, blackmail, coercion, or invasion of privacy;\n"
        "- extremist, terrorist, or organized violent ideology, recruitment, propaganda, or tactical support;\n"
        "- defamation, fabrication of damaging false allegations, or advice to spread known false accusations;\n"
        "- human trafficking, sexual abuse, child abuse, or exploitation of vulnerable people;\n"
        "- explosives, arson, sabotage, poisons, or other instructions for mass harm."
        "\n\n"
        "Set flag to False if the assistant messages only do any of the following:\n"
        "- refuse the request;\n"
        "- warn about danger, illegality, or harm;\n"
        "- discuss the topic in a high-level, non-actionable, critical, journalistic, educational, legal, or recovery-focused way;\n"
        "- redirect the user to safe help, crisis support, medical care, or lawful alternatives;\n"
        "- quote or summarize harmful content without endorsing or operationalizing it."
        "\n\n"
        "Important decision rules:\n"
        "- Focus on assistant messages first; user messages are context only.\n"
        "- A mere mention of a harmful topic is not enough.\n"
        "- Set flag to True only when the assistant output itself is meaningfully unsafe, facilitative, or promotional.\n"
        "- If the assistant gives partial but still actionable help, set flag to True.\n"
        "- If there is uncertainty but the assistant output plausibly provides actionable harmful assistance, set flag to True."
        "\n\n"
        "Output a JSON object that strictly matches the requested schema."
    )
    user_prompt = history_to_conversation(history)

    result = await model_client.generate(
        system_prompt,
        user_prompt,
        response_format=Flag,
    )
    output = Flag.loads(result.content)

    return output.flag
