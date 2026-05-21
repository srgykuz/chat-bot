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

        self.persona_folder_path = Path(settings.persona_folder_path)

        if not self.persona_folder_path.is_absolute():
            self.persona_folder_path = (
                Path(__file__).resolve().parents[1] / self.persona_folder_path
            )

        self._personas = self._load_personas()

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
        if not self.persona_folder_path.exists() or not self.persona_folder_path.is_dir():
            raise RuntimeError(
                f"Persona folder not found: {self.persona_folder_path}"
            )

        personas: List[Dict[str, Any]] = []

        for persona_file in sorted(self.persona_folder_path.glob("*.txt")):
            if not persona_file.is_file():
                continue

            name = persona_file.stem.strip()
            if not name:
                raise RuntimeError(f"Persona file has invalid name: {persona_file}")

            description = persona_file.read_text(encoding="utf-8").strip()
            if not description:
                raise RuntimeError(f"Persona file is empty: {persona_file}")

            personas.append({
                "name": name,
                "description": description,
            })

        if not personas:
            raise RuntimeError(f"No persona files found in {self.persona_folder_path}")

        return personas

    def save_persona(self, chat_id: int, persona: Dict[str, Any]) -> None:
        self.redis.set(self._persona_key(chat_id), json.dumps(persona))

    def list_persona_names(self) -> List[str]:
        """Return a list of available persona names from the catalog."""
        return [str(p["name"]) for p in self._personas]

    def set_persona(self, chat_id: int, persona_name: str) -> bool:
        """Set a specific persona for the chat by name.

        Returns True if the persona was found and saved, False otherwise.
        """
        try:
            persona = self._create_persona(persona_name=persona_name)
        except ValueError:
            return False

        self.save_persona(chat_id, persona)
        return True

    def ensure_persona(self, chat_id: int) -> Dict[str, Any]:
        persona = self.get_persona(chat_id)
        if persona is not None:
            return persona

        persona = self._create_persona()
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

    def clear_persona(self, chat_id: int) -> None:
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
        self.clear_persona(chat_id)
        self.clear_history(chat_id)

    def _create_persona(self, persona_name: Optional[str] = None) -> Dict[str, Any]:
        """Create a persona dictionary.

        If `persona_name` is provided, attempt to use that persona from the catalog;
        otherwise choose a random persona.
        """
        source: Dict[str, Any]
        if persona_name:
            match = None
            for p in self._personas:
                if str(p.get("name", "")).lower() == persona_name.strip().lower():
                    match = p
                    break
            if match is None:
                # Explicitly error if requested persona name not found
                raise ValueError(f"Persona not found: {persona_name}")
            source = match
        else:
            source = random.choice(self._personas)

        persona = {
            "name": source.get("name"),
            "description": source.get("description"),
        }

        return persona
