"""Shared bot logic for processing Telegram updates."""
import logging
from typing import Dict, Any, Optional
from src.llm import ModelClient
from src.session import Message, MessageRole, Session, Persona, User
from src.telegram import TelegramClient, TelegramMessage, parse_update

logger = logging.getLogger(__name__)

telegram_client = TelegramClient()
session_store = Session()
model_client = ModelClient()


async def process_update(update: Dict[str, Any]) -> Optional[TelegramMessage]:
    """Process a Telegram update and respond to the user."""
    message_info = parse_update(update)
    if not message_info:
        logger.debug("Update did not contain text message, skipping")
        return None

    chat_id = message_info.chat_id
    if chat_id is None:
        logger.debug("Update did not contain a chat id, skipping")
        return None

    if message_info.text is None:
        logger.debug("Update did not contain text, skipping")
        return None

    update_id = message_info.update_id
    text = message_info.text.strip()
    logger.info(f"Processing update {update_id} from {message_info.username}: {text}")

    if text.startswith("/"):
        response_text = await _handle_command(chat_id, text)
        await telegram_client.send_message(
            chat_id=chat_id,
            text=response_text,
            reply_to_message_id=message_info.message_id,
            escape=False
        )
        logger.info(f"Processed command for update {update_id} from {message_info.username}")
        return message_info

    await telegram_client.send_chat_action(chat_id, action="typing")

    session_store.append_history(chat_id, Message(role=MessageRole.USER, content=text))

    persona = session_store.init_persona(chat_id)
    history = session_store.get_history(chat_id)
    user = User(
        first_name=message_info.first_name,
        last_name=message_info.last_name,
    )

    try:
        system_prompt = model_client.build_system_prompt(persona, user)
        response_text = await model_client.chat(system_prompt, history)
        session_store.append_history(chat_id, Message(role=MessageRole.ASSISTANT, content=response_text))
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc, exc_info=True)
        response_text = "Sorry, I couldn't think of a good answer right now. Let's keep talking!"

    await telegram_client.send_message(
        chat_id=chat_id,
        text=response_text
    )

    logger.info(f"Sent response for update {update_id} to {message_info.username}")
    return message_info


async def _handle_command(chat_id: int, text: str) -> str:
    command = text.split()[0].lower()

    if command == "/get_persona":
        persona = session_store.get_persona(chat_id)
        if not persona:
            return "No persona is currently saved for this chat."

        return (
            "*Current persona:*\n"
            f"Name: `{persona.name}`\n"
        )

    if command == "/clear_persona":
        session_store.clear(chat_id)
        return "Persona cleared. A new persona will be created on the next message."

    if command == "/set_persona":
        parts = text.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            return "Usage: /set\\_persona <Name>"

        name = parts[1].strip()
        persona: Optional[Persona] = None

        try:
            persona = session_store.select_persona(name)
        except Exception:
            persona = None

        if persona:
            session_store.set_persona(chat_id, persona)
            return f"Persona set to {persona.name}."
        else:
            return f"Persona {name} not found."

    if command == "/list_persona":
        names = [p.name for p in session_store.personas]
        if not names:
            return "No personas available."
        names = [f"`{n}`" for n in names]
        return "*Available personas:*\n" + "\n".join(names)

    if command == "/get_history":
        info = session_store.get_history_info(chat_id)
        return (
            "*Chat history info:*\n"
            f"Total messages: `{info.num_messages}`\n"
            f"Max history stored: `{info.max_messages}`\n"
            f"User messages: `{info.num_user_messages}`\n"
            f"Assistant messages: `{info.num_assistant_messages}`"
        )

    if command == "/clear_history":
        session_store.clear(chat_id)
        return "Chat history cleared."

    if command == "/clear":
        session_store.clear(chat_id)
        return "Session cleared: persona and history removed."

    return (
        "Persona commands:\n"
        "/set\\_persona <Name>\n"
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
