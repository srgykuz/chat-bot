"""Shared bot logic for processing Telegram updates."""
import logging
from typing import Dict, Any, Optional
from src.telegram_handler import TelegramHandler, TelegramUpdateParser

logger = logging.getLogger(__name__)

telegram_handler = TelegramHandler()
update_parser = TelegramUpdateParser()


async def process_update(update: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a Telegram update and respond to the user."""
    message_info = update_parser.parse_update(update)
    if not message_info:
        logger.debug("Update did not contain text message, skipping")
        return None

    logger.info(f"Processing message from {message_info['username']}: {message_info['text']}")
    response_text = f"Echo: {message_info['text']}"

    await telegram_handler.send_message(
        chat_id=message_info["chat_id"],
        text=response_text,
        reply_to_message_id=message_info["message_id"]
    )

    logger.info(f"Sent response to {message_info['username']}")
    return message_info
