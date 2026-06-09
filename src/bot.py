import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import timedelta

from src.llm import ModelClient
from src.session import Message, MessageRole, SessionClient, Persona, User
from src.weather import fetch_weather, WeatherInfo
from src.telegram import TelegramClient, TelegramMessage, parse_update
from src.config import get_settings, get_queue


logger = logging.getLogger(__name__)
settings = get_settings()
queue = get_queue()
telegram_client = TelegramClient()
session_client = SessionClient()
model_client = ModelClient()


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
    response = ""

    if command == "/get_persona":
        persona = session_client.get_persona(chat_id)

        if persona:
            response = (
                "*Current persona:*\n"
                f"ID: `{persona.id}`\n"
                f"Name: `{persona.name}`\n"
            )
        else:
            response = "No persona is currently selected for this chat."
    elif command == "/clear_persona":
        session_client.clear(chat_id)
        response = "Persona cleared. A new persona will be created on the next message."
    elif command == "/set_persona":
        parts = text.split(maxsplit=1)

        if len(parts) < 2 or not parts[1].strip():
            response = "Usage: /set\\_persona <id>"
        else:
            persona_id = parts[1].strip()
            persona: Optional[Persona] = None

            try:
                persona = session_client.select_persona(persona_id)
            except Exception:
                persona = None

            if persona:
                session_client.set_persona(chat_id, persona)
                response = f"Persona set to {persona.id}."
            else:
                response = f"Persona {persona_id} not found."
    elif command == "/list_persona":
        ids = [p.id for p in session_client.load_personas()]

        if ids:
            ids = [f"`{n}`" for n in ids]
            response = "*Available personas:*\n" + "\n".join(ids)
        else:
            response = "No personas available."
    elif command == "/get_history":
        info = session_client.get_history_info(chat_id)
        response = (
            "*Chat history info:*\n"
            f"Total messages: `{info.num_messages}`\n"
            f"Max history stored: `{info.max_messages}`\n"
            f"User messages: `{info.num_user_messages}`\n"
            f"Assistant messages: `{info.num_assistant_messages}`"
        )
    elif command == "/clear_history":
        session_client.clear(chat_id)
        response = "Chat history cleared."
    elif command == "/clear":
        session_client.clear(chat_id)
        response = "Session cleared."
    else:
        response = (
            "Persona commands:\n"
            "/set\\_persona <id>\n"
            "/get\\_persona\n"
            "/list\\_persona\n"
            "/clear\\_persona\n"
            "\n"
            "History commands:\n"
            "/get\\_history\n"
            "/clear\\_history\n"
            "\n"
            "Other commands:\n"
            "/clear"
        )

    await telegram_client.send_message(
        chat_id=chat_id,
        text=response,
        reply_to_message_id=message.message_id,
        escape=False
    )


async def handle_message(message: TelegramMessage) -> None:
    """
    Handles a message that contain plain text a LLM should respond to in the chat context.
    """
    chat_id = message.chat_id

    if not chat_id:
        return

    text = (message.text or "").strip()

    if not text:
        return

    token = session_client.buffer_message(chat_id, message)
    enqueue_flush_buffered_messages(chat_id, token)

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
    input = []

    for msg in messages:
        if msg.chat_id != chat_id:
            raise ValueError(f"Message chat_id {msg.chat_id} does not match target chat_id {chat_id}")

        text = (msg.text or "").strip()

        if text:
            input.append(text)

    if not input:
        logger.info(f"No messages to process for chat {chat_id}")
        return

    history = session_client.get_history(chat_id)

    for text in input:
        history.append(Message(role=MessageRole.USER, content=text))

    persona = session_client.init_persona(chat_id)
    user = User(
        first_name=messages[-1].first_name,
        last_name=messages[-1].last_name,
    )
    persona_weather: Optional[WeatherInfo] = None

    try:
        persona_weather = await fetch_weather(persona.city, lang="ru")
    except Exception as e:
        logger.error(f"Error fetching weather info: {e}")

    system_prompt = model_client.build_system_prompt(persona, user, persona_weather)
    response = ""
    success = False

    try:
        response = await model_client.chat(system_prompt, history)
        success = True
    except Exception as e:
        response = "🤖"
        success = False
        logger.error("LLM call error: %s", e, exc_info=True)

    output = response.split(settings.output_separator)
    output = [s.strip() for s in output if s.strip()]

    if success:
        for text in input:
            session_client.append_history(chat_id, Message(role=MessageRole.USER, content=text))

        for text in output:
            session_client.append_history(chat_id, Message(role=MessageRole.ASSISTANT, content=text))

    logger.info(f"Responding to chat {chat_id} from {messages[-1].username}: {response}")

    for text in output:
        await telegram_client.send_chat_action(chat_id, action="typing")

        delay = calc_typing_duration(text)
        await asyncio.sleep(delay)

        await telegram_client.send_message(chat_id=chat_id, text=text)


def calc_typing_duration(text: str) -> float:
    """
    Returns number of seconds to simulate human typing duration for a given text.
    """
    chars_per_second = 15

    return len(text) / chars_per_second


def enqueue_flush_buffered_messages(chat_id: int, token: str) -> None:
    """
    Schedules execution of `flush_buffered_messages()`.
    """
    job_id = f"flush_buffered_messages_{chat_id}_{token}"
    execute_in = timedelta(seconds=settings.chat_flush_interval)

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
