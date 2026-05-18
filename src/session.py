"""Session storage and persona management for Friend Bot."""
import json
import hashlib
from typing import Any, Dict, List, Optional, Union, cast
from redis import Redis
from src.config import get_settings


RedisValue = Union[str, bytes]


class SessionStore:
    """Manage per-user conversation history and persona storage."""

    MAX_HISTORY_MESSAGES = 12

    def __init__(self) -> None:
        settings = get_settings()
        self.redis = Redis.from_url(settings.redis_url, decode_responses=True)

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

    def save_persona(self, chat_id: int, persona: Dict[str, Any]) -> None:
        self.redis.set(self._persona_key(chat_id), json.dumps(persona))

    def ensure_persona(self, chat_id: int, user_name: Optional[str] = None) -> Dict[str, Any]:
        persona = self.get_persona(chat_id)
        if persona is not None:
            return persona

        persona = self._create_persona(chat_id, user_name)
        self.save_persona(chat_id, persona)
        return persona

    def append_message(self, chat_id: int, role: str, content: str) -> None:
        message = {"role": role, "content": content}
        self.redis.rpush(self._history_key(chat_id), json.dumps(message))
        self.redis.ltrim(self._history_key(chat_id), -self.MAX_HISTORY_MESSAGES, -1)

    def get_history(self, chat_id: int) -> List[Dict[str, Any]]:
        raw_items = cast(List[RedisValue], self.redis.lrange(self._history_key(chat_id), -self.MAX_HISTORY_MESSAGES, -1))
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

    def _create_persona(self, chat_id: int, user_name: Optional[str]) -> Dict[str, Any]:
        names = ["Anna", "Max", "Mira", "Leo", "Sasha", "Noa", "Jamie", "Finn"]
        tones = [
            "warm and witty",
            "calm and supportive",
            "thoughtful and curious",
            "playful and encouraging",
        ]
        hobbies = [
            "reading new novels",
            "exploring local cafes",
            "drawing small sketches",
            "watching indie movies",
            "trying new recipes",
            "going for evening walks",
        ]
        name = names[chat_id % len(names)]
        tone = tones[chat_id % len(tones)]
        hobby = hobbies[chat_id % len(hobbies)]
        user_hint = f" and likes talking to {user_name}" if user_name else ""

        description = (
            f"{name} is a {tone} virtual friend who listens closely and responds like a real person. "
            f"They enjoy {hobby}{user_hint}. "
            "They keep the conversation human, do not mention they are an AI, and they remember the most recent topics."
        )

        seed = hashlib.sha256(str(chat_id).encode()).hexdigest()[:8]
        identity = f"{name}#{seed}"

        return {
            "name": name,
            "identity": identity,
            "tone": tone,
            "hobby": hobby,
            "description": description,
        }
