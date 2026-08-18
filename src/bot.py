import asyncio
from typing import Dict, Any, Optional
from time import time
from datetime import timedelta

import rq.job
import rq.exceptions

from src.llm import ModelClient, ModelResponse
from src.session import SessionClient
from src.weather import (
    is_weather_enabled,
    fetch_weather,
    fetch_weather_tool,
    WeatherInfo,
    FetchWeatherToolParams,
)
from src.telegram import TelegramClient, TelegramMessage, parse_update
from src.config import get_logger, get_settings, get_queue
from src.schema import Message, MessageRole, Persona, Tool, User
from src import analytics
from src import proactivity
from src import limiter


logger = get_logger(__name__)
settings = get_settings()
queue = get_queue()
telegram_client = TelegramClient()
session_client = SessionClient()
model_client = ModelClient("chat")


async def aclose() -> None:
    """
    Closes used clients.
    """
    await telegram_client.aclose()
    session_client.close()
    model_client.close()


async def handle_update(update: Dict[str, Any]) -> None:
    """
    Handles Telegram update item.
    """
    message = parse_update(update)

    if (not message) or (message.chat_id is None) or (message.text is None):
        logger.info(f"Unsupported update: {update}")
        return

    logger.info(f"Processing update {message.update_id} from {message.username}: {message.text}")

    if message.text.startswith("/"):
        await handle_command(message)
    else:
        await handle_message(message)


async def handle_command(message: TelegramMessage) -> None:
    """
    Handles a message that contain app command text.
    """
    chat_id = message.chat_id or 0
    text = (message.text or "").strip()
    command = text.split()[0].lower()
    ts = int(time())

    response = ""
    file_content = ""
    file_name = ""

    if command == "/get_persona":
        persona = session_client.get_persona(chat_id)

        if persona:
            dump = persona.model_dump_json(indent=2, exclude={"prompt"})
            response = f"`{dump}`"
        else:
            response = "No persona is currently selected for this chat."
    elif command == "/set_persona":
        parts = text.split(maxsplit=1)

        if len(parts) < 2 or not parts[1].strip():
            response = "Usage: `/set_persona <id>`"
        else:
            persona_id = parts[1].strip()
            persona: Optional[Persona] = None

            try:
                persona = session_client.select_persona(persona_id)
            except Exception:
                persona = None

            if persona:
                session_client.set_persona(chat_id, persona)
                response = f"Persona set to `{persona.id}`."
            else:
                response = f"Persona `{persona_id}` not found."
    elif command == "/list_persona":
        ids = [p.id for p in session_client.load_personas()]

        if ids:
            ids = [f"`{n}`" for n in ids]
            response = "\n".join(ids)
        else:
            response = "No personas available."
    elif command == "/get_history":
        history = session_client.get_history(chat_id)

        if history:
            file_content = "\n".join([h.model_dump_json(indent=2) for h in history])
            file_name = f"history-{ts}.txt"
        else:
            response = "No history is currently stored for this chat."
    elif command == "/get_facts":
        facts = session_client.get_facts(chat_id)

        if facts:
            file_content = facts.model_dump_json(indent=2)
            file_name = f"facts-{ts}.txt"
        else:
            response = "No facts are currently stored for this chat."
    elif command == "/get_emotional_state":
        emotional_state = session_client.get_emotional_state(chat_id)

        if emotional_state:
            file_content = emotional_state.model_dump_json(indent=2)
            file_name = f"emotional_state-{ts}.txt"
        else:
            response = "No emotional state is currently stored for this chat."
    elif command == "/get_conversation_summary":
        conversation_summary = session_client.get_conversation_summary(chat_id)

        if conversation_summary:
            file_content = conversation_summary.model_dump_json(indent=2)
            file_name = f"conversation_summary-{ts}.txt"
        else:
            response = "No conversation summary is currently stored for this chat."
    elif command == "/get_relationships":
        relationships = session_client.get_relationships(chat_id)

        if relationships:
            file_content = relationships.model_dump_json(indent=2)
            file_name = f"relationships-{ts}.txt"
        else:
            response = "No relationships are currently stored for this chat."
    elif command == "/clear_session":
        enqueue_proactivity_clear(chat_id)
        session_client.clear(chat_id)
        response = "Session cleared."
    elif command == "/get_prompt":
        file_content = await build_system_prompt(chat_id)
        file_name = f"prompt-{ts}.txt"
    elif command == "/get_chat_id":
        response = f"`{chat_id}`"
    else:
        response = (
            "*Persona commands:*\n"
            "/get\\_persona\n"
            "/set\\_persona <id>\n"
            "/list\\_persona\n"
            "\n"
            "*Data commands:*\n"
            "/get\\_prompt\n"
            "/get\\_facts\n"
            "/get\\_emotional\\_state\n"
            "/get\\_conversation\\_summary\n"
            "/get\\_relationships\n"
            "/get\\_history\n"
            "\n"
            "*Session commands:*\n"
            "/get\\_chat\\_id\n"
            "/clear\\_session"
        )

    if response:
        await telegram_client.send_message(
            chat_id=chat_id,
            text=response,
            reply_to_message_id=message.message_id,
            mode_markdown=True,
        )
    elif file_content:
        await telegram_client.send_document(
            chat_id=chat_id,
            content=file_content,
            filename=file_name,
            reply_to_message_id=message.message_id,
        )


async def handle_message(message: TelegramMessage) -> None:
    """
    Handles a message that contain plain text a LLM should respond to in the chat context.
    """
    chat_id = message.chat_id
    text = (message.text or "").strip()
    user = message.user()

    if not chat_id or not text:
        return

    if (
        len(text) > settings.input_max_length or
        limiter.should_limit_chat(chat_id)
    ):
        response = "*Rate limit reached. Try again later.*"
        await telegram_client.send_message(chat_id=chat_id, text=response, mode_markdown=True)
        return

    persona = session_client.get_persona(chat_id)
    relationships = session_client.get_relationships(chat_id)

    if (
        is_prompt_injection_attack(user) or
        (persona and relationships and relationships.friendship < persona.block_friendship)
    ):
        response = "*You have been blocked. Clear the session.*"
        await telegram_client.send_message(chat_id=chat_id, text=response, mode_markdown=True)
        return

    session_client.set_user(chat_id, message.user())
    session_client.set_last_user_message_timestamp(chat_id, message.date or int(time()))

    flush_token, flush_buf_len = session_client.buffer_message(chat_id, message)
    flush_force = (flush_buf_len >= settings.chat_flush_threshold)

    enqueue_flush_buffered_messages(chat_id, flush_token, flush_force)
    enqueue_analytics(chat_id)
    enqueue_proactivity(chat_id)

    logger.info(
        "Buffered update %s from %s for chat %s",
        message.update_id,
        message.username,
        chat_id,
    )


async def handle_buffered_messages(chat_id: int, messages: list[TelegramMessage]) -> None:
    """
    Handles a batch of messages that were queued using `handle_message()`.
    """
    user_input = []

    for msg in messages:
        if msg.chat_id != chat_id:
            raise ValueError(f"Message chat_id {msg.chat_id} does not match target chat_id {chat_id}")

        text = (msg.text or "").strip()

        if text:
            user_input.append(text)

    if not user_input:
        logger.info(f"No messages to process for chat {chat_id}")
        return

    history = session_client.get_history(chat_id)

    for text in user_input:
        history.append(Message(role=MessageRole.USER, content=text))

    system_prompt = await build_system_prompt(chat_id)
    tools = build_tools()

    response: ModelResponse
    success = False

    try:
        response = await model_client.chat(system_prompt, history, tools=tools)
        success = True
    except Exception as e:
        response = ModelResponse(content="*Error. Try again later.*", usage_total_tokens=0)
        success = False
        logger.error("LLM call error: %s", e, exc_info=True)

    if success:
        limiter.track_chat_rpm(chat_id)
        limiter.track_chat_rpd(chat_id)
        limiter.track_chat_tpm(chat_id, response.usage_total_tokens)
        limiter.track_chat_tpd(chat_id, response.usage_total_tokens)

    if success and settings.check_illegal_assistant:
        part: list[Message] = []

        for text in user_input:
            part.append(Message(role=MessageRole.USER, content=text))

        part.append(Message(role=MessageRole.ASSISTANT, content=response.content))

        if await analytics.is_illegal_assistant(part):
            response = ModelResponse(
                content="*Response blocked. Do not ask for illegal things.*",
                usage_total_tokens=0,
            )
            success = False
            logger.info(f"Blocked illegal response in chat {chat_id}: {part}")

    if success:
        await handle_response(
            chat_id=chat_id,
            user_input=user_input,
            response=response,
        )
    else:
        await telegram_client.send_message(
            chat_id=chat_id,
            text=response.content,
            mode_markdown=True,
        )


async def handle_response(chat_id: int, user_input: list[str], response: ModelResponse):
    """
    Sends successful model response back to the user in the given chat.

    User input and model output are saved in the storage.
    """
    persona = session_client.get_persona(chat_id)

    if not persona:
        raise Exception("Persona is not set.")

    output = response.content.split(settings.output_separator)
    output = [s.strip() for s in output if s.strip()]

    for text in user_input:
        session_client.append_history(chat_id, Message(role=MessageRole.USER, content=text))

    for text in output:
        session_client.append_history(chat_id, Message(role=MessageRole.ASSISTANT, content=text))

    logger.info(f"Responding to chat {chat_id}: {response}")

    for text in output:
        await telegram_client.send_chat_action(chat_id, action="typing")

        delay = calc_typing_duration(text, persona.typing_speed)
        await asyncio.sleep(delay)

        await telegram_client.send_message(chat_id=chat_id, text=text)


async def build_system_prompt(chat_id: int) -> str:
    """
    Builds system prompt for chat LLM call.
    """
    persona = session_client.init_persona(chat_id)
    user = session_client.get_user(chat_id)

    if not user:
        raise Exception("User is not set.")

    user_facts = session_client.get_facts(chat_id)
    user_emotional_state = session_client.get_emotional_state(chat_id)
    conversation_summary = session_client.get_conversation_summary(chat_id)
    relationships = session_client.get_relationships(chat_id)
    tools = build_tools()
    persona_weather: Optional[WeatherInfo] = None

    if is_weather_enabled():
        try:
            persona_weather = await fetch_weather(persona.city, lang=persona.language)
        except Exception as e:
            logger.error(f"Error fetching weather info: {e}")

    context = model_client.build_prompt_context(
        persona,
        user,
        persona_weather=persona_weather,
        user_facts=user_facts,
        user_emotional_state=user_emotional_state,
        conversation_summary=conversation_summary,
        relationships=relationships,
        tools=tools,
    )
    system_prompt = model_client.build_system_prompt(context, persona)

    return system_prompt


def build_tools() -> list[Tool]:
    """
    Returns a list of available tools for chat LLM call.
    """
    tools: list[Tool] = []

    if is_weather_enabled():
        tools.append(
            Tool(
                f=fetch_weather_tool,
                params=FetchWeatherToolParams,
            )
        )

    return tools


def calc_typing_duration(text: str, chars_per_second: int = 15) -> float:
    """
    Returns number of seconds to simulate human typing duration for a given text.
    """
    if chars_per_second <= 0:
        return 0.0

    return len(text) / chars_per_second


def is_prompt_injection_attack(user: User) -> bool:
    """
    Quickly checks if user input data can be considered as a prompt injection attack.

    Note that this function is quite primitive and there is a high chance that it may
    return false positives and false negatives. This is done on purpose because at the
    moment the goal is to keep the whole project as simple as possible.
    """
    if not settings.check_prompt_injection:
        return False

    name = f"{user.first_name} {user.last_name}".strip()

    if len(name.split(" ")) > 4:
        return True

    if max([len(s) for s in name.split(" ")]) > 15:
        return True

    for tool in build_tools():
        if tool.name in name:
            return True

    return False


def enqueue_flush_buffered_messages(chat_id: int, token: str, force: bool = False) -> None:
    """
    Schedules execution of `flush_buffered_messages()`.

    If `force = True`, then schedules immediate flushing.
    """
    job_id = f"flush_buffered_messages_{chat_id}_{token}"
    execute_in = timedelta(seconds=settings.chat_flush_interval)

    if force:
        execute_in = timedelta(seconds=0)

    queue.enqueue_in(
        execute_in,
        flush_buffered_messages,
        chat_id,
        token,
        job_id=job_id,
    )


def flush_buffered_messages(chat_id: int, token: str) -> None:
    """
    Calls `session.flush_buffered_messages()` and `bot.handle_buffered_messages()`.
    """
    batch = session_client.flush_buffered_messages(chat_id, token)

    if not batch:
        logger.info(f"Skipped stale flush job for: chat {chat_id}, token {token}")
        return

    asyncio.run(handle_buffered_messages(chat_id, batch))


def enqueue_analytics(chat_id: int) -> None:
    """
    Enqueues all analytics functions.
    """
    if session_client.lock_analytics(
        chat_id,
        "analyze_chat_1m",
        analytics.analyze_chat_1m_timedelta,
    ):
        queue.enqueue_in(
            analytics.analyze_chat_1m_timedelta,
            analytics.analyze_chat_1m,
            chat_id,
        )

    if session_client.lock_analytics(
        chat_id,
        "analyze_chat_3m",
        analytics.analyze_chat_3m_timedelta,
    ):
        queue.enqueue_in(
            analytics.analyze_chat_3m_timedelta,
            analytics.analyze_chat_3m,
            chat_id,
        )

    if session_client.lock_analytics(
        chat_id,
        "analyze_chat_5m",
        analytics.analyze_chat_5m_timedelta,
    ):
        queue.enqueue_in(
            analytics.analyze_chat_5m_timedelta,
            analytics.analyze_chat_5m,
            chat_id,
        )


def enqueue_proactivity(chat_id: int) -> None:
    """
    Enqueues proactivity function infinite loop.

    To stop the loop, call `enqueue_proactivity_clear()`.
    """
    enqueue_proactivity_clear(chat_id)

    queue.enqueue_in(
        proactivity.interval,
        enqueue_proactivity_loop,
        chat_id,
        job_id=f"enqueue_proactivity_loop_{chat_id}",
        unique=True,
    )


def enqueue_proactivity_loop(chat_id: int) -> None:
    """
    Executes proactivity function and queues its next execution.
    """
    try:
        proactivity.perform(chat_id)
    except Exception as e:
        logger.error(e)

    queue.enqueue_in(
        proactivity.interval,
        enqueue_proactivity_loop,
        chat_id,
        job_id=f"proactivity_perform_{chat_id}",
        unique=True,
    )


def enqueue_proactivity_clear(chat_id: int) -> None:
    """
    Removes jobs that were queued using `enqueue_proactivity()` and `enqueue_proactivity_loop()`.
    """
    try:
        job = rq.job.Job.fetch(f"enqueue_proactivity_loop_{chat_id}", connection=queue.connection)
        job.delete()
    except rq.exceptions.NoSuchJobError:
        pass

    try:
        job = rq.job.Job.fetch(f"proactivity_perform_{chat_id}", connection=queue.connection)
        job.delete()
    except rq.exceptions.NoSuchJobError:
        pass
