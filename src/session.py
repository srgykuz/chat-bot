import json
import random
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Any, cast
from enum import StrEnum

import yaml
from redis import Redis

from src.config import get_settings


class MessageRole(StrEnum):
    """
    Who created a message.
    """
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(frozen=True, slots=True)
class Message:
    """
    A message in the conversation history.
    """
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        role = MessageRole(data.get("role", ""))
        content = str(data.get("content", ""))
        timestamp = float(data.get("timestamp", 0.0))

        return cls(
            role=role,
            content=content,
            timestamp=timestamp,
        )


@dataclass(frozen=True, slots=True)
class Persona:
    """
    Information about a persona that we are mimicking in the conversation.
    """
    id: str
    name: str
    timezone: str
    city: str
    prompt: str

    def is_valid(self) -> bool:
        return bool(self.id and self.name and self.timezone and self.city and self.prompt)


@dataclass(frozen=True, slots=True)
class User:
    """
    Information about a user that we are talking to.
    """
    first_name: Optional[str]
    last_name: Optional[str]

    def country(self) -> Optional[str]:
        return "Россия"


@dataclass(frozen=True, slots=True)
class HistoryInfo:
    """
    Meta information about the conversation history.
    """
    max_messages: int
    num_messages: int
    num_user_messages: int
    num_assistant_messages: int


class SessionClient:
    """
    Manages per-user and per-chat state: history, persona, etc.
    A data in the state must not be considered as permanent as it
    may be lost at any time. The state is stored in Redis.
    """
    def __init__(self) -> None:
        self.settings = get_settings()
        self.redis = Redis.from_url(self.settings.redis_url, decode_responses=True)

    def close(self) -> None:
        """
        Closes the underlying resources.
        """
        self.redis.close()

    def clear(self, chat_id: int) -> None:
        """
        Removes entire state for the given chat ID.
        """
        pipe = self.redis.pipeline()

        pipe.delete(self._history_key(chat_id))
        pipe.delete(self._persona_key(chat_id))

        pipe.execute()

    def _persona_key(self, chat_id: int) -> str:
        """
        Returns the Redis key for storing the persona of a specific chat.
        """
        return f"session:{chat_id}:persona"

    def _history_key(self, chat_id: int) -> str:
        """
        Returns the Redis key for storing the conversation history of a specific chat.
        """
        return f"session:{chat_id}:history"

    def load_personas(self) -> List[Persona]:
        """
        Loads persona definitions from the specified directory.

        Each persona should be defined in a separate directory using files:
        params.yml and prompt.md. Empty personas are ignored.
        """
        dir = Path(self.settings.personas_path)

        if not dir.exists() or not dir.is_dir():
            raise RuntimeError(f"Persona directory not found: {dir}")

        personas: List[Persona] = []

        for persona_dir in sorted(dir.iterdir()):
            if not persona_dir.is_dir():
                continue

            params_path = persona_dir / "params.yml"

            if not params_path.exists() or not params_path.is_file():
                continue

            params_raw = params_path.read_text(encoding="utf-8")
            params = yaml.safe_load(params_raw)

            if not isinstance(params, dict):
                raise RuntimeError(f"Persona params file must contain a YAML object: {params_path}")

            id = str(params.get("id", None) or "").strip()
            name = str(params.get("name", None) or "").strip()
            timezone = str(params.get("timezone", None) or "").strip()
            city = str(params.get("city", None) or "").strip()

            prompt_path = persona_dir / "prompt.md"

            if not prompt_path.exists() or not prompt_path.is_file():
                continue

            prompt = prompt_path.read_text(encoding="utf-8").strip()

            if not prompt:
                continue

            persona = Persona(
                id=id,
                name=name,
                timezone=timezone,
                city=city,
                prompt=prompt,
            )

            if not persona.is_valid():
                raise RuntimeError(f"Persona definition is invalid: {persona_dir}")

            personas.append(persona)

        if not personas:
            raise RuntimeError(f"No personas found in the catalog: {self.settings.personas_path}")

        return personas

    def get_persona(self, chat_id: int) -> Optional[Persona]:
        """
        Returns the currently set persona for the given chat ID, or None if no persona is set.
        """
        key = self._persona_key(chat_id)
        persona_id = cast(Optional[str], self.redis.get(key))

        if persona_id is None:
            return None

        personas = self.load_personas()

        for persona in personas:
            if persona.id == persona_id:
                return persona

        return None

    def set_persona(self, chat_id: int, persona: Persona) -> None:
        """
        Sets the given persona for the specified chat ID.
        """
        key = self._persona_key(chat_id)

        self.redis.set(key, persona.id)

    def select_persona(self, persona_id: Optional[str] = None) -> Persona:
        """
        Selects a persona from the catalog.

        If persona_id is provided, tries to find a persona with that id (case-insensitive).
        If no persona is found, raises an exception. If persona_id is not provided, then
        selects a random persona from the catalog.
        """
        personas = self.load_personas()
        persona_id = persona_id.strip() if persona_id else None

        if not persona_id:
            return random.choice(personas)

        persona: Optional[Persona] = None

        for p in personas:
            if p.id.casefold() == persona_id.casefold():
                persona = p
                break

        if not persona:
            raise ValueError(f"Persona not found: {persona_id}")

        return persona

    def init_persona(self, chat_id: int) -> Persona:
        """
        Ensures that a persona is set for the given chat ID.
        If no persona is set, a new one will be created and set.

        Returns the currently set or newly created persona.
        """
        existing_persona = self.get_persona(chat_id)

        if existing_persona is not None:
            return existing_persona

        new_persona = self.select_persona()
        self.set_persona(chat_id, new_persona)

        return new_persona

    def get_history(self, chat_id: int) -> List[Message]:
        """
        Returns the conversation history for the given chat ID.
        Ordered from oldest to newest message.
        """
        key = self._history_key(chat_id)
        items = cast(List[str], self.redis.lrange(key, -self.settings.history_limit, -1))
        history: List[Message] = []

        for item in items:
            if not item:
                continue

            data = json.loads(item)
            msg = Message.from_dict(data)
            history.append(msg)

        return history

    def append_history(self, chat_id: int, message: Message) -> None:
        """
        Appends a message to the conversation history for the given chat ID.
        The conversation history is trimmed to the maximum length defined in the settings.
        """
        key = self._history_key(chat_id)
        value = json.dumps(message.to_dict())
        pipe = self.redis.pipeline()

        pipe.rpush(key, value)
        pipe.ltrim(key, -self.settings.history_limit, -1)

        pipe.execute()

    def get_history_info(self, chat_id: int) -> HistoryInfo:
        """
        Returns meta information about the conversation history for the given chat ID.
        """
        history = self.get_history(chat_id)
        num_user = sum(1 for msg in history if msg.role == MessageRole.USER)
        num_assistant = sum(1 for msg in history if msg.role == MessageRole.ASSISTANT)

        return HistoryInfo(
            max_messages=self.settings.history_limit,
            num_messages=len(history),
            num_user_messages=num_user,
            num_assistant_messages=num_assistant,
        )
