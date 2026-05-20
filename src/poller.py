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
        self.pending_tasks: list[asyncio.Task[Dict | None]] = []

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

                self._cleanup_done_tasks()

                updates = payload.get("result", []) or []
                if not updates:
                    continue

                latest_id = max(
                    (update.get("update_id") for update in updates if update.get("update_id") is not None),
                    default=None,
                )
                if latest_id is not None:
                    self.offset = latest_id + 1

                for update in updates:
                    task = asyncio.create_task(process_update(update))
                    self.pending_tasks.append(task)

            except asyncio.CancelledError:
                logger.info("Telegram polling task cancelled")
                break
            except Exception as exc:
                logger.error("Telegram polling error: %s", exc, exc_info=True)
                await asyncio.sleep(5)

    def _cleanup_done_tasks(self) -> None:
        """Remove completed tasks and log exceptions."""
        still_pending: list[asyncio.Task[Dict | None]] = []
        for task in self.pending_tasks:
            if task.done():
                exc = task.exception()
                if exc is not None:
                    logger.error("Error handling update task: %s", exc, exc_info=True)
            else:
                still_pending.append(task)
        self.pending_tasks = still_pending


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(TelegramPoller().start())
