import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, cast
from enum import StrEnum

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

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Persona:
    """
    Information about a persona that we are mimicking in the conversation.
    """
    name: str
    prompt: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

    def tz_offset(self) -> int:
        return 3


@dataclass(frozen=True, slots=True)
class User:
    """
    Information about a user that we are talking to.
    """
    first_name: Optional[str]
    last_name: Optional[str]

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

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

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


class Session:
    """
    Manages per-user and per-chat state: history, persona, etc.
    A data in the state must not be considered as permanent as it
    may be lost at any time. The state is stored in Redis.
    """
    def __init__(self) -> None:
        self.settings = get_settings()
        self.redis = Redis.from_url(self.settings.redis_url, decode_responses=True)
        self.personas = self.load_personas(Path(self.settings.persona_dir_path))

        if not self.personas:
            raise RuntimeError(f"No personas found in the catalog: {self.settings.persona_dir_path}")

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

    def load_personas(self, dir: Path) -> List[Persona]:
        """
        Loads persona prompts from the specified directory.
        Empty personas are ignored.

        Each persona should be defined in a separate .txt file, where the filename
        (without extension) is the persona name, and the file content is the persona prompt.
        """
        if not dir.exists() or not dir.is_dir():
            raise RuntimeError(f"Persona directory not found: {dir}")

        personas: List[Persona] = []

        for file in sorted(dir.glob("*.txt")):
            if not file.is_file():
                continue

            name = file.stem.strip()

            if not name:
                raise RuntimeError(f"Persona file has invalid name: {file}")

            prompt = file.read_text(encoding="utf-8").strip()

            if not prompt:
                continue

            personas.append(Persona(name=name, prompt=prompt))

        return personas

    def get_persona(self, chat_id: int) -> Optional[Persona]:
        """
        Returns the currently set persona for the given chat ID, or None if no persona is set.
        """
        key = self._persona_key(chat_id)
        raw = cast(Optional[str], self.redis.get(key))

        if raw is None:
            return None

        data = json.loads(raw)
        persona = Persona(**data)

        return persona

    def set_persona(self, chat_id: int, persona: Persona) -> None:
        """
        Sets the given persona for the specified chat ID.
        """
        key = self._persona_key(chat_id)
        value = json.dumps(persona.to_dict())

        self.redis.set(key, value)

    def select_persona(self, persona_name: Optional[str] = None) -> Persona:
        """
        Selects a persona from the catalog.

        If persona_name is provided, tries to find a persona with that name (case-insensitive).
        If no persona is found, raises an exception. If persona_name is not provided, then
        selects a random persona from the catalog.
        """
        persona_name = persona_name.strip() if persona_name else None

        if not persona_name:
            return random.choice(self.personas)

        persona: Optional[Persona] = None

        for p in self.personas:
            if p.name.casefold() == persona_name.casefold():
                persona = p
                break

        if not persona:
            raise ValueError(f"Persona not found: {persona_name}")

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
        items = cast(List[str], self.redis.lrange(key, -self.settings.max_history_messages, -1))
        history: List[Message] = []

        for item in items:
            if not item:
                continue

            data = json.loads(item)
            history.append(Message(**data))

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
        pipe.ltrim(key, -self.settings.max_history_messages, -1)

        pipe.execute()

    def get_history_info(self, chat_id: int) -> HistoryInfo:
        """
        Returns meta information about the conversation history for the given chat ID.
        """
        history = self.get_history(chat_id)
        num_user = sum(1 for msg in history if msg.role == MessageRole.USER)
        num_assistant = sum(1 for msg in history if msg.role == MessageRole.ASSISTANT)

        return HistoryInfo(
            max_messages=self.settings.max_history_messages,
            num_messages=len(history),
            num_user_messages=num_user,
            num_assistant_messages=num_assistant,
        )
