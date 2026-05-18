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
    logger.info(f"Processing message from {message_info['username']}: {message_info['text']}")

    persona = session_store.ensure_persona(chat_id, user_name=message_info.get("first_name"))
    session_store.append_message(chat_id, "user", message_info["text"])
    history = session_store.get_history(chat_id)

    try:
        response_text = await llm_client.chat_with_friend(persona, history)
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc, exc_info=True)
        response_text = "Sorry, I couldn't think of a good answer right now. Let's keep talking!"

    session_store.append_message(chat_id, "assistant", response_text)

    await telegram_handler.send_message(
        chat_id=chat_id,
        text=response_text,
        reply_to_message_id=message_info["message_id"]
    )

    logger.info(f"Sent response to {message_info['username']}")
    return message_info
