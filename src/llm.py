"""LLM integration for generating friend-like responses."""
import asyncio
import logging
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper for LLM provider calls."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = None
        self.openai = None
        self.system_prompt_template_path = Path(self.settings.system_prompt_template_path)
        self.llm_params_path = Path(self.settings.llm_params_path)

        if not self.system_prompt_template_path.is_absolute():
            self.system_prompt_template_path = (
                Path(__file__).resolve().parents[1] / self.system_prompt_template_path
            )

        if not self.llm_params_path.is_absolute():
            self.llm_params_path = Path(__file__).resolve().parents[1] / self.llm_params_path

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
        template = self._load_system_prompt_template()
        mapping = {
            "current_time": datetime.now(tz=timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S"),
            **persona
        }
        try:
            return template.format_map(mapping)
        except KeyError as exc:
            raise RuntimeError(
                f"System prompt template is missing persona field: {exc.args[0]}"
            ) from exc

    def _load_system_prompt_template(self) -> str:
        if not self.system_prompt_template_path.exists():
            raise RuntimeError(
                f"System prompt template not found: {self.system_prompt_template_path}"
            )
        return self.system_prompt_template_path.read_text(encoding="utf-8")

    def _load_llm_params(self) -> Dict[str, Any]:
        try:
            if not self.llm_params_path.exists():
                raise RuntimeError(f"LLM params file not found: {self.llm_params_path}")

            raw = self.llm_params_path.read_text(encoding="utf-8")
            data = json.loads(raw)

            if not isinstance(data, dict):
                raise RuntimeError("LLM params file must contain a JSON object at the top level.")

            return data
        except Exception:
            raise RuntimeError(f"Error occurred while loading LLM params from {self.llm_params_path}")

    async def _openai_chat(self, messages: List[Dict[str, str]]) -> str:
        return await asyncio.to_thread(self._openai_call, messages)

    def _openai_call(self, messages: List[Dict[str, str]]) -> str:
        if not self.openai:
            raise RuntimeError("OpenAI client is not initialized.")

        params = self._load_llm_params()

        # build kwargs for the OpenAI call, only include known keys
        allowed_keys = {
            "model",
            "temperature",
            "max_completion_tokens",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "reasoning_effort",
            "verbosity",
        }
        call_kwargs: Dict[str, Any] = {}

        for k in allowed_keys:
            if k not in params:
                continue

            if params[k] is None:
                continue  # skip null values

            if isinstance(params[k], str) and not params[k].strip():
                continue  # skip empty string values

            call_kwargs[k] = params[k]

        # messages handled separately
        call_kwargs["messages"] = messages
        completion = self.openai.chat.completions.create(**call_kwargs)
        output = (completion.choices[0].message.content or "").strip()

        call_kwargs.pop("messages", None)  # remove messages from kwargs for cleaner logging
        logger.info(
            "OpenAI call model=%s args=%s usage=%s",
            getattr(completion, "model", None),
            call_kwargs,
            getattr(completion, "usage", None),
        )

        return output
