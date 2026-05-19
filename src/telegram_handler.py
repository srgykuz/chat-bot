"""Telegram API handler for sending and receiving messages."""
import httpx
from typing import Optional, Dict, Any
from src.config import get_settings


class TelegramHandler:
    """Handler for Telegram API interactions."""

    def __init__(self):
        self.settings = get_settings()
        self.token = self.settings.telegram_token
        self.api_url = f"https://api.telegram.org/bot{self.token}"
        self.client = httpx.AsyncClient()

    async def send_message(self, chat_id: int, text: str, reply_to_message_id: Optional[int] = None) -> Dict[str, Any]:
        """Send a message to a Telegram chat.

        Args:
            chat_id: Telegram chat ID
            text: Message text
            reply_to_message_id: Optional message ID to reply to

        Returns:
            API response
        """
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }

        if reply_to_message_id:
            payload["reply_to_message_id"] = reply_to_message_id

        try:
            response = await self.client.post(
                f"{self.api_url}/sendMessage",
                json=payload,
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            print(f"Error sending message: {e}")
            raise

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> Dict[str, Any]:
        """Send a chat action to Telegram, such as typing."""
        try:
            response = await self.client.post(
                f"{self.api_url}/sendChatAction",
                json={"chat_id": chat_id, "action": action},
                timeout=10.0,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            print(f"Error sending chat action: {e}")
            raise

    async def set_webhook(self, url: str) -> Dict[str, Any]:
        """Set webhook URL for receiving updates.

        Args:
            url: Webhook URL

        Returns:
            API response
        """
        try:
            response = await self.client.post(
                f"{self.api_url}/setWebhook",
                json={"url": url},
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            print(f"Error setting webhook: {e}")
            raise

    async def get_webhook_info(self) -> Dict[str, Any]:
        """Get current webhook info.

        Returns:
            Webhook info
        """
        try:
            response = await self.client.get(
                f"{self.api_url}/getWebhookInfo",
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            print(f"Error getting webhook info: {e}")
            raise

    async def get_updates(self, offset: Optional[int] = None, timeout: int = 30, limit: int = 100) -> Dict[str, Any]:
        """Retrieve updates for long polling.

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

        try:
            response = await self.client.post(
                f"{self.api_url}/getUpdates",
                json=payload,
                timeout=timeout + 10.0,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            print(f"Error fetching updates: {e}")
            raise


class TelegramUpdateParser:
    """Parser for Telegram updates."""

    @staticmethod
    def parse_update(update: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a Telegram update to extract message info.

        Args:
            update: Raw update from Telegram

        Returns:
            Parsed message info or None if not a message update
        """
        if "message" not in update:
            return None

        message = update["message"]

        if "text" not in message:
            return None

        return {
            "update_id": update.get("update_id"),
            "chat_id": message.get("chat", {}).get("id"),
            "message_id": message.get("message_id"),
            "user_id": message.get("from", {}).get("id"),
            "username": message.get("from", {}).get("username"),
            "first_name": message.get("from", {}).get("first_name"),
            "text": message.get("text"),
            "date": message.get("date"),
        }
