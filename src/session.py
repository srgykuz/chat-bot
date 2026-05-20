"""Session storage and persona management for Friend Bot."""
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast
from redis import Redis
from src.config import get_settings


RedisValue = Union[str, bytes]


class SessionStore:
    """Manage per-user conversation history and persona storage."""

    def __init__(self) -> None:
        settings = get_settings()
        self.redis = Redis.from_url(settings.redis_url, decode_responses=True)
        self.max_history_messages = settings.max_history_messages

        self.persona_catalog_path = Path(settings.persona_catalog_path)
        self.persona_template_path = Path(settings.persona_template_path)

        if not self.persona_catalog_path.is_absolute():
            self.persona_catalog_path = (
                Path(__file__).resolve().parents[1] / self.persona_catalog_path
            )
        if not self.persona_template_path.is_absolute():
            self.persona_template_path = (
                Path(__file__).resolve().parents[1] / self.persona_template_path
            )

        self._personas = self._load_personas()
        self._persona_template = self._load_persona_template()

    def _persona_key(self, chat_id: int) -> str:
        return f"session:{chat_id}:persona"

    def _history_key(self, chat_id: int) -> str:
        return f"session:{chat_id}:history"

    def get_persona(self, chat_id: int) -> Optional[Dict[str, Any]]:
        raw = cast(Optional[RedisValue], self.redis.get(self._persona_key(chat_id)))
        if raw is None:
            return None

        if isinstance(raw, bytes):
            raw = raw.decode()

        return json.loads(raw)

    def _load_personas(self) -> List[Dict[str, Any]]:
        if not self.persona_catalog_path.exists():
            raise RuntimeError(
                f"Persona catalog not found: {self.persona_catalog_path}"
            )

        raw = self.persona_catalog_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, list):
            raise RuntimeError("Persona catalog must be a JSON list of persona objects.")

        return data

    def _load_persona_template(self) -> str:
        if not self.persona_template_path.exists():
            raise RuntimeError(
                f"Persona template not found: {self.persona_template_path}"
            )
        return self.persona_template_path.read_text(encoding="utf-8")

    def save_persona(self, chat_id: int, persona: Dict[str, Any]) -> None:
        self.redis.set(self._persona_key(chat_id), json.dumps(persona))

    def ensure_persona(self, chat_id: int, user_name: Optional[str] = None) -> Dict[str, Any]:
        persona = self.get_persona(chat_id)
        if persona is not None:
            return persona

        persona = self._create_persona(user_name)
        self.save_persona(chat_id, persona)
        return persona

    def append_message(self, chat_id: int, role: str, content: str) -> None:
        message = {"role": role, "content": content}
        self.redis.rpush(self._history_key(chat_id), json.dumps(message))
        self.redis.ltrim(self._history_key(chat_id), -self.max_history_messages, -1)

    def get_history(self, chat_id: int) -> List[Dict[str, Any]]:
        raw_items = cast(List[RedisValue], self.redis.lrange(self._history_key(chat_id), -self.max_history_messages, -1))
        history: List[Dict[str, Any]] = []

        for item in raw_items:
            if not item:
                continue
            if isinstance(item, bytes):
                item = item.decode()
            history.append(json.loads(item))

        return history

    def clear_history(self, chat_id: int) -> None:
        self.redis.delete(self._history_key(chat_id))

    def delete_persona(self, chat_id: int) -> None:
        self.redis.delete(self._persona_key(chat_id))

    def get_history_info(self, chat_id: int) -> Dict[str, int]:
        history = self.get_history(chat_id)
        num_user = sum(1 for message in history if message.get("role") == "user")
        num_assistant = sum(1 for message in history if message.get("role") == "assistant")

        return {
            "num_messages": len(history),
            "max_history_messages": self.max_history_messages,
            "num_user_messages": num_user,
            "num_assistant_messages": num_assistant,
        }

    def clear(self, chat_id: int) -> None:
        self.delete_persona(chat_id)
        self.clear_history(chat_id)

    def _create_persona(self, user_name: Optional[str]) -> Dict[str, Any]:
        persona = random.choice(self._personas)
        persona = {
            "name": persona.get("name"),
            "tone": persona.get("tone"),
            "hobby": persona.get("hobby"),
        }
        persona["user_hint"] = f" and likes talking to {user_name}" if user_name else ""
        persona["description"] = self._persona_template.format_map(persona)

        return persona
