import asyncio
import random
from datetime import timedelta
from time import time
from typing import Optional

from src.config import get_logger
from src.session import SessionClient
from src.llm import ModelClient, ModelResponse, history_to_conversation
from src.schema import (
    MessageRole,
    ProactivityMessage,
    ProactivityState,
    ProactivityAction,
)


logger = get_logger(__name__)
session_client = SessionClient()
model_client = ModelClient("proactivity")

interval = timedelta(hours=1)


def close() -> None:
    """
    Closes underlying resources.
    """
    session_client.close()
    model_client.close()


def perform(chat_id: int) -> None:
    """
    Generates "proactive message" based on the current chat state and
    sends it to a user.

    "proactive message" is a message initiated by assistant rather than user.
    The assistant either continues existing topic or starts a new one.

    This function intended to be executed on defined interval.
    """
    persona = session_client.get_persona(chat_id)
    user = session_client.get_user(chat_id)
    relationships = session_client.get_relationships(chat_id)
    history = session_client.get_history(chat_id)
    last_user_ts = session_client.get_last_user_message_timestamp(chat_id)
    state = session_client.get_proactivity_state(chat_id)

    if not (persona and user and history and last_user_ts):
        logger.info("Chat state is not initialized")
        return

    if (persona.proactivity_factor == 0) or persona.is_sleeping():
        return

    if relationships and relationships.friendship < persona.proactivity_friendship:
        return

    if (state is None) or (state.last_user_ts != last_user_ts):
        state = ProactivityState(
            last_user_ts=last_user_ts,
            last_action=ProactivityAction.Nothing,
            follow_up_count=0,
            daily_event_count=0,
            ping_count=0,
        )

    context = model_client.build_prompt_context(
        persona,
        user,
        persona_weather=None,
        user_facts=None,
        user_emotional_state=None,
        conversation_summary=None,
        relationships=relationships,
        tools=None,
    )
    persona_prompt = model_client.build_persona_prompt(context, persona)
    conversation = history_to_conversation(history)

    now = time()
    delta = timedelta(seconds=(now - last_user_ts))

    result: Optional[ProactivityMessage] = None

    if delta < timedelta(hours=3):
        pass
    elif delta < timedelta(hours=12):
        if state.follow_up_count >= 3:
            pass
        elif state.last_action == ProactivityAction.FollowUp or history[-1].role == MessageRole.USER:
            if persona.proactivity_factor >= random.random():
                result = generate_continuation(persona_prompt, conversation)
                state.follow_up_count += 1
            else:
                pass
        else:
            if persona.proactivity_factor >= random.random():
                result = generate_follow_up(persona_prompt, conversation)
                state.last_action = ProactivityAction.FollowUp
                state.follow_up_count += 1
            else:
                pass
    elif delta < timedelta(hours=30):
        if state.daily_event_count >= 2:
            pass
        elif state.last_action == ProactivityAction.DailyEvent:
            if persona.proactivity_factor >= random.random():
                result = generate_continuation(persona_prompt, conversation)
                state.daily_event_count += 1
            else:
                pass
        else:
            if persona.proactivity_factor >= random.random():
                result = generate_daily_event(persona_prompt)
                state.last_action = ProactivityAction.DailyEvent
                state.daily_event_count += 1
            else:
                pass
    elif delta > timedelta(hours=48):
        if state.ping_count >= 1:
            pass
        else:
            if persona.proactivity_factor >= random.random():
                result = generate_ping(persona_prompt, conversation)
                state.last_action = ProactivityAction.Ping
                state.ping_count += 1
            else:
                pass

    session_client.set_proactivity_state(chat_id, state)

    if result and result.message:
        from src.bot import handle_response

        response = ModelResponse(
            content=result.message,
            usage_total_tokens=0
        )

        asyncio.run(handle_response(
            chat_id=chat_id,
            user_input=[],
            response=response,
        ))


def generate_follow_up(persona_prompt: str, conversation: str) -> ProactivityMessage:
    """
    Generates "follow-up message": a message that mentions something from the
    conversation history. Intended to be sent after a short pause in the dialog.
    """
    system_prompt = (
        "You are a backend engine. Review the provided conversation history between "
        "user and assistant. Identify whether there is an ongoing or unresolved topic "
        "that makes a bot-initiated follow-up message feel relevant after a pause in "
        "the conversation. Write one follow-up message about it in the user's language. "
        "Output must strictly match the requested JSON schema."
        "\n\n"
        "The follow-up message style should match your persona character and writing style:"
        "\n\n"
        f"{persona_prompt}"
    )
    user_prompt = conversation

    response = asyncio.run(model_client.generate(
        system_prompt,
        user_prompt,
        response_format=ProactivityMessage,
    ))
    result = ProactivityMessage.loads(response.content)

    return result


def generate_ping(persona_prompt: str, conversation: str) -> ProactivityMessage:
    """
    Generates "ping message": a message that aims on getting answer from a user.
    Intended to be sent after a long pause in the dialog.
    """
    system_prompt = (
        "You are a backend engine. You is chatting with a user on behalf of a person. "
        "The user is keeping silence for a long time already. Compose a message to write "
        "first. Start a new topic, ask what's up or mention something from the conversation "
        "history. The message should be short and should gently nudge the user to reply "
        "after a long pause. The message should fit persona's style and language. You will "
        "receive the conversation history between the user and you. Output must strictly "
        "match the requested JSON schema."
        "\n\n"
        "Persona description:"
        "\n\n"
        f"{persona_prompt}"
    )
    user_prompt = conversation

    response = asyncio.run(model_client.generate(
        system_prompt,
        user_prompt,
        response_format=ProactivityMessage,
    ))
    result = ProactivityMessage.loads(response.content)

    return result


def generate_daily_event(persona_prompt: str) -> ProactivityMessage:
    """
    Generates "daily event message": a message describing daily event happened to persona.
    Intended to be sent occasionally, after a short pause in the dialog.
    """
    topics = [
        "missed train / delayed commute",
        "coffee shop mix-up",
        "buying groceries after work",
        "phone battery died at the wrong time",
        "small task at a post office or bank",
        "rainy weather changed plans",
        "found a lost item and returned it",
        "awkward but harmless elevator interaction",
        "a package arrived damaged",
        "ran into an old acquaintance",
        "cooking mistake at home",
        "forgot an umbrella",
        "traffic jam on the way home",
        "dog or cat causing a small mess",
        "a meeting ran longer than expected",
        "tried a new restaurant or snack",
        "working out after a long break",
        "laptop / app / Wi-Fi problem",
        "cleaning the apartment and found something old",
        "bought something unnecessary but useful",
        "stayed late at the office",
        "helped a neighbor with something small",
        "took the wrong bus or exit",
        "had a funny conversation with a cashier",
        "got stuck waiting for someone",
        "made a simple weekend plan",
        "received an unexpected message",
        "visited a pharmacy or clinic for a minor reason",
        "noticed a small change in the neighborhood",
        "did a boring errand that turned mildly interesting",
    ]
    system_prompt = (
        "You are a backend engine. Invent one believable daily event that could "
        "have happened to a real human today. Make it specific enough to sound "
        "natural. Add extra details and context. This event should fit as a topic "
        "for daily conversation between friends. The event should happen from a "
        "third-person perspective. Participant should be mentioned as 'persona'. "
        "Output must strictly match the requested JSON schema."
    )
    user_prompt = (
        "Generate one natural everyday event about: "
        f"{random.choice(topics)}"
    )

    response = asyncio.run(model_client.generate(
        system_prompt,
        user_prompt,
        response_format=ProactivityMessage,
    ))
    event = ProactivityMessage.loads(response.content)

    system_prompt = (
        "You are a backend engine. Paraphrase the event to make it look like a "
        "chat message. This message is initiated by persona. The message starts "
        "conversation between persona and it friend after a short pause. Form "
        "the message according to the persona's style and language. Output must "
        "strictly match the requested JSON schema."
        "\n\n"
        "Persona description:"
        "\n\n"
        f"{persona_prompt}"
    )
    user_prompt = (
        "The event that happened to the persona: "
        f"{event.message}"
    )

    response = asyncio.run(model_client.generate(
        system_prompt,
        user_prompt,
        response_format=ProactivityMessage,
    ))
    result = ProactivityMessage.loads(response.content)

    return result


def generate_continuation(persona_prompt: str, conversation: str) -> ProactivityMessage:
    """
    Generates "continuation message": a message that continues dialog starting from the
    last assistant messages. Intended to be sent after a short pause in the dialog.
    """
    system_prompt = (
        "You are a backend engine. Read the latest messages in the conversation "
        "between user and assistant and write one continuation message on behalf of "
        "assistant that naturally follows from the same topic and context. This is "
        "not a new topic, not a check-in, and not a follow-up after silence. "
        "It should feel like the very next message in the same ongoing exchange. "
        "Output must strictly match the requested JSON schema."
        "\n\n"
        "The message style should match your persona character, writing style and language:"
        "\n\n"
        f"{persona_prompt}"
    )
    user_prompt = conversation

    response = asyncio.run(model_client.generate(
        system_prompt,
        user_prompt,
        response_format=ProactivityMessage,
    ))
    result = ProactivityMessage.loads(response.content)

    return result
