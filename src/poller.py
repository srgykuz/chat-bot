"""Telegram long polling runner for development."""
import asyncio
import logging
from typing import Optional, Dict, Any
from src.config import get_settings
from src.telegram_handler import TelegramHandler
from src.bot import process_update

logger = logging.getLogger(__name__)

settings = get_settings()


class TelegramPoller:
    """Telegram long polling runner."""

    def __init__(self) -> None:
        self.handler = TelegramHandler()
        self.offset: Optional[int] = None
        self.timeout = 30
        self.limit = 100

    async def start(self) -> None:
        """Start polling and handling updates."""
        logger.info("Telegram long polling started")

        while True:
            try:
                payload = await self.handler.get_updates(
                    offset=self.offset,
                    timeout=self.timeout,
                    limit=self.limit,
                )

                updates = payload.get("result", []) or []
                if not updates:
                    continue

                for update in updates:
                    await process_update(update)
                    update_id = update.get("update_id")
                    if update_id is not None:
                        self.offset = update_id + 1

            except asyncio.CancelledError:
                logger.info("Telegram polling task cancelled")
                break
            except Exception as exc:
                logger.error("Telegram polling error: %s", exc, exc_info=True)
                await asyncio.sleep(5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(TelegramPoller().start())
