"""LLM integration for generating friend-like responses."""
import asyncio
import logging
from typing import Any, Dict, List, Optional
from src.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper for LLM provider calls."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = None
        self.openai = None

        if self.settings.openai_api_key:
            try:
                import openai
            except ImportError as exc:
                raise RuntimeError("OpenAI SDK is not installed. Add openai to requirements.") from exc

            self.openai = openai.OpenAI(api_key=self.settings.openai_api_key)
            self.provider = "openai"

        elif self.settings.gemini_api_key:
            raise RuntimeError("Gemini support is not implemented yet. Please use OPENAI_API_KEY for now.")

    async def chat_with_friend(self, persona: Dict[str, Any], history: List[Dict[str, str]]) -> str:
        if self.provider != "openai":
            raise RuntimeError("No LLM provider configured. Set OPENAI_API_KEY.")

        messages = [
            {
                "role": "system",
                "content": self._build_system_prompt(persona),
            }
        ]
        messages.extend(history)

        return await self._openai_chat(messages)

    def _build_system_prompt(self, persona: Dict[str, Any]) -> str:
        return (
            f"You are {persona['name']}, a {persona['tone']} virtual friend. "
            f"{persona['description']}"
        )

    async def _openai_chat(self, messages: List[Dict[str, str]]) -> str:
        return await asyncio.to_thread(self._openai_call, messages)

    def _openai_call(self, messages: List[Dict[str, str]]) -> str:
        if not self.openai:
            raise RuntimeError("OpenAI client is not initialized.")

        completion = self.openai.chat.completions.create(
            model="gpt-5.4",
            messages=messages, # type: ignore
            temperature=0.8,
            max_completion_tokens=500,
            top_p=0.9,
            presence_penalty=0.2,
            frequency_penalty=0.2,
        )
        return (completion.choices[0].message.content or "").strip()
