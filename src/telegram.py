import asyncio
from dataclasses import dataclass
import logging
from typing import Any, Callable, Coroutine, Dict, Optional

import httpx

from src.config import get_settings


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TelegramMessage:
    """
    https://core.telegram.org/bots/api#message
    """
    update_id: Optional[int]
    message_id: Optional[int]
    chat_id: Optional[int]
    user_id: Optional[int]
    username: Optional[str]
    first_name: Optional[str]
    last_name: Optional[str]
    text: Optional[str]
    date: Optional[int]


def parse_update(update: Dict[str, Any]) -> Optional[TelegramMessage]:
    """
    Parse a Telegram update to extract text message info.
    If the update does not contain a text message, returns None.
    """
    message = update.get("message")

    if (not message) or ("text" not in message):
        return None

    chat = message.get("chat", {})
    fromm = message.get("from", {})

    return TelegramMessage(
        update_id=update.get("update_id"),
        message_id=message.get("message_id"),
        chat_id=chat.get("id"),
        user_id=fromm.get("id"),
        username=fromm.get("username"),
        first_name=fromm.get("first_name"),
        last_name=fromm.get("last_name"),
        text=message.get("text"),
        date=message.get("date"),
    )


class TelegramClient:
    """
    Client for interaction with the Telegram Bot API.
    """
    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = f"https://api.telegram.org/bot{self.settings.telegram_token}"
        self.client = httpx.AsyncClient()

    async def __aenter__(self) -> "TelegramClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """
        Closes the underlying HTTP client session.
        """
        await self.client.aclose()

    def _log_http_error(self, error: Exception, request: object) -> None:
        """
        Log an httpx error with request payload and response details.
        """
        response = getattr(error, "response", None)
        response_text = None

        if response is not None:
            response_text = response.text

        status = getattr(response, "status_code", None)

        logger.error(
            "Telegram API error: %s; request=%s; response_status=%s; response_text=%s",
            error,
            request,
            status,
            response_text,
            exc_info=True,
        )

    def _escape_markdown(self, text: str) -> str:
        """
        Escape Markdown special characters supported by Telegram.
        """
        for char in ("_", "*", "`", "["):
            text = text.replace(char, f"\\{char}")

        return text

    async def _request_json(
        self,
        http_method: str,
        api_method: str,
        payload: Dict[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Makes an HTTP request to the Telegram Bot API and returns the JSON response.
        """
        url = f"{self.base_url}/{api_method}"
        request_kwargs: Dict[str, Any] = {
            "timeout": timeout,
        }

        if payload is not None:
            request_kwargs["json"] = payload

        try:
            response = await self.client.request(http_method, url, **request_kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as error:
            self._log_http_error(error, {"method": http_method, "url": url, "payload": payload})
            raise

    async def send_message(
        self,
        chat_id: int,
        text: str,
        reply_to_message_id: Optional[int] = None,
        escape: bool = True,
    ) -> Dict[str, Any]:
        """
        Send a message to a Telegram chat.

        Args:
            chat_id: Telegram chat ID
            text: Message text
            reply_to_message_id: Optional message ID to reply to
            escape: Whether to escape Markdown special characters

        Returns:
            API response
        """
        if escape:
            text = self._escape_markdown(text)

        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }

        if reply_to_message_id:
            payload["reply_to_message_id"] = reply_to_message_id

        return await self._request_json("POST", "sendMessage", payload=payload)

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> Dict[str, Any]:
        """
        Sets the chat action for a Telegram chat (e.g., "typing").
        """
        payload = {
            "chat_id": chat_id,
            "action": action
        }

        return await self._request_json("POST", "sendChatAction", payload=payload)

    async def set_webhook(self, url: str) -> Dict[str, Any]:
        """
        Set webhook URL for receiving updates.
        """
        payload = {"url": url}

        return await self._request_json("POST", "setWebhook", payload=payload)

    async def get_webhook_info(self) -> Dict[str, Any]:
        """
        Get current webhook info.
        """
        return await self._request_json("GET", "getWebhookInfo")

    async def get_updates(
        self,
        offset: Optional[int] = None,
        timeout: int = 30,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Retrieve updates for long polling.

        Args:
            offset: Identifier of the first update to be returned.
            timeout: Timeout in seconds for the long polling request.
            limit: Maximum number of updates to retrieve.

        Returns:
            API response payload.
        """
        payload = {
            "timeout": timeout,
            "limit": limit,
        }

        if offset is not None:
            payload["offset"] = offset

        # Add buffer to avoid client timeout
        request_timeout = timeout + 10

        return await self._request_json("POST", "getUpdates", payload=payload, timeout=request_timeout)


class TelegramPoller:
    """
    Client for long polling Telegram updates and handling them asynchronously.
    """
    def __init__(
        self,
        handler: Callable[[Dict[str, Any]], Coroutine[Any, Any, Any]],
    ) -> None:
        self.client = TelegramClient()
        self.handler = handler
        self.offset: Optional[int] = None
        self.pending_tasks: list[asyncio.Task[Any]] = []

    async def aclose(self) -> None:
        """
        Close resources owned by the poller.
        """
        await self.client.aclose()

    async def start(self) -> None:
        """
        Starts polling and handling updates.

        Polling continues until the task is cancelled.
        Updates are processed asynchronously.
        Each update is processed by the provided handler function.
        """
        logger.info("Telegram polling started")

        while True:
            try:
                self._cleanup_completed_tasks()

                response = await self.client.get_updates(self.offset)
                updates = response.get("result", [])

                if not updates:
                    continue

                latest_id = max(update.get("update_id", 0) for update in updates)

                if latest_id != 0:
                    self.offset = latest_id + 1

                for update in updates:
                    task = asyncio.create_task(self.handler(update))
                    self.pending_tasks.append(task)
            except asyncio.CancelledError:
                logger.info("Telegram polling cancelled")
                break
            except Exception as exc:
                logger.error("Telegram polling error: %s", exc, exc_info=True)
                await asyncio.sleep(5)

    def _cleanup_completed_tasks(self) -> None:
        """
        Clears completed tasks and logs any exceptions that occurred during their handling.
        """
        still_pending: list[asyncio.Task[Any]] = []

        for task in self.pending_tasks:
            if task.done():
                exc = task.exception()

                if exc is not None:
                    logger.error("Telegram polling task error: %s", exc, exc_info=True)
            else:
                still_pending.append(task)

        self.pending_tasks = still_pending
