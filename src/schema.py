from enum import StrEnum
from typing import Self, Optional
from time import time

from pydantic import BaseModel, Field


class BaseModelJSON(BaseModel):
    def dumps(self) -> str:
        return self.model_dump_json()

    @classmethod
    def loads(cls, s: str) -> Self:
        return cls.model_validate_json(s)


class User(BaseModelJSON):
    first_name: str
    last_name: str = Field(default="")


class Persona(BaseModelJSON):
    id: str
    name: str
    timezone: str
    city: str
    prompt: str


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModelJSON):
    role: MessageRole
    content: str
    timestamp: float = Field(default_factory=time)


class HistoryInfo(BaseModelJSON):
    max_messages: int
    num_messages: int
    num_user_messages: int
    num_assistant_messages: int


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


class Fact(BaseModelJSON):
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
