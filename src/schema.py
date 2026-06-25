from enum import StrEnum
from typing import Self, Optional, Any
from time import time
from dataclasses import dataclass, asdict, field

from pydantic import BaseModel, Field


@dataclass(frozen=True, slots=True)
class User:
    """
    Information about a user that we are talking to.
    """
    first_name: Optional[str]
    last_name: Optional[str]


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
    timestamp: float = field(default_factory=time)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        role = MessageRole(data.get("role", ""))
        content = str(data.get("content", ""))
        timestamp = float(data.get("timestamp", 0.0))

        return cls(
            role=role,
            content=content,
            timestamp=timestamp,
        )


@dataclass(frozen=True, slots=True)
class HistoryInfo:
    """
    Meta information about the conversation history.
    """
    max_messages: int
    num_messages: int
    num_user_messages: int
    num_assistant_messages: int


class BaseModelJSON(BaseModel):
    def dumps(self) -> str:
        return self.model_dump_json()

    @classmethod
    def loads(cls, s: str) -> Self:
        return cls.model_validate_json(s)


class Mood(StrEnum):
    CHEERFUL = "cheerful"
    CALM = "calm"
    SAD = "sad"
    ANGRY = "angry"
    FLIRTY = "flirty"


class Tone(StrEnum):
    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    RUDE = "rude"
    ROMANTIC = "romantic"


class Engagement(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class EmotionalStateLLM(BaseModelJSON):
    mood: Mood
    tone: Tone
    engagement: Engagement
    mood_confidence: float = Field(..., ge=0.0, le=1.0)
    tone_confidence: float = Field(..., ge=0.0, le=1.0)
    engagement_confidence: float = Field(..., ge=0.0, le=1.0)


class EmotionalState(EmotionalStateLLM):
    timestamp: float = Field(default_factory=time)


class FactTag(StrEnum):
    HOBBY = "hobby"
    JOB = "job"
    FOOD = "food"
    MUSIC = "music"
    GAME = "game"
    MOVIE = "movie"
    BOOK = "book"
    LIKE = "like"
    DISLIKE = "dislike"


class Fact(BaseModel):
    tag: FactTag
    value: str


class Facts(BaseModelJSON):
    facts: list[Fact]


class ConversationSummaryLLM(BaseModelJSON):
    summaries: list[str]


class ConversationSummary(ConversationSummaryLLM):
    timestamps: list[float]

    def to_llm(self) -> ConversationSummaryLLM:
        return ConversationSummaryLLM.model_validate(self.model_dump())
