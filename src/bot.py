"""Shared bot logic for processing Telegram updates."""
import logging
from typing import Dict, Any, Optional
from src.llm import LLMClient
from src.session import SessionStore
from src.telegram_handler import TelegramHandler, TelegramUpdateParser

logger = logging.getLogger(__name__)

telegram_handler = TelegramHandler()
update_parser = TelegramUpdateParser()
session_store = SessionStore()
llm_client = LLMClient()


async def process_update(update: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a Telegram update and respond to the user."""
    message_info = update_parser.parse_update(update)
    if not message_info:
        logger.debug("Update did not contain text message, skipping")
        return None

    chat_id = message_info["chat_id"]
    text = message_info["text"].strip()
    logger.info(f"Processing message from {message_info['username']}: {text}")

    if text.startswith("/"):
        response_text = await _handle_command(chat_id, text, message_info["first_name"])
        await telegram_handler.send_message(
            chat_id=chat_id,
            text=response_text,
            reply_to_message_id=message_info["message_id"],
        )
        logger.info(f"Processed command for {message_info['username']}")
        return message_info

    await telegram_handler.send_chat_action(chat_id, action="typing")

    persona = session_store.ensure_persona(chat_id, user_name=message_info.get("first_name"))
    session_store.append_message(chat_id, "user", text)
    history = session_store.get_history(chat_id)

    try:
        response_text = await llm_client.chat_with_friend(persona, history)
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc, exc_info=True)
        response_text = "Sorry, I couldn't think of a good answer right now. Let's keep talking!"

    session_store.append_message(chat_id, "assistant", response_text)

    await telegram_handler.send_message(
        chat_id=chat_id,
        text=response_text
    )

    logger.info(f"Sent response to {message_info['username']}")
    return message_info


async def _handle_command(chat_id: int, text: str, user_name: Optional[str]) -> str:
    command = text.split()[0].lower()

    if command == "/get_persona":
        persona = session_store.get_persona(chat_id)
        if not persona:
            return "No persona is currently saved for this chat."

        return (
            "*Current persona:*\n"
            f"Name: `{persona.get('name')}`\n"
            f"Tone: `{persona.get('tone')}`\n"
            f"Hobby: `{persona.get('hobby')}`\n"
            f"Description: `{persona.get('description')}`"
        )

    if command == "/clear_persona":
        session_store.clear_persona(chat_id)
        return "Persona cleared. A new persona will be created on the next message."

    if command == "/get_history":
        info = session_store.get_history_info(chat_id)
        return (
            "*Chat history info:*\n"
            f"Total messages: `{info['num_messages']}`\n"
            f"Max history stored: `{info['max_history_messages']}`\n"
            f"User messages: `{info['num_user_messages']}`\n"
            f"Assistant messages: `{info['num_assistant_messages']}`"
        )

    if command == "/clear_history":
        session_store.clear_history(chat_id)
        return "Chat history cleared."

    if command == "/clear":
        session_store.clear(chat_id)
        return "Session cleared: persona and history removed."

    return (
        "Available commands:\n"
        "/get\\_persona\n"
        "/clear\\_persona\n"
        "/get\\_history\n"
        "/clear\\_history\n"
        "/clear"
    )
