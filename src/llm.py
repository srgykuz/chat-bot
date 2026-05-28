"""LLM integration for generating friend-like responses."""
import asyncio
import logging
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from google.genai.types import ModelContent, Part, UserContent, GenerateContentConfig
from src.config import get_settings
from src.session import Message, Persona

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper for LLM provider calls."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = None
        self.openai = None
        self.gemini = None
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
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError("Gemini SDK is not installed. Add google-genai to requirements.") from exc

            self.gemini = genai.Client(api_key=self.settings.gemini_api_key)
            self.provider = "gemini"

        logger.info("LLMClient initialized with provider: %s", self.provider)

    async def chat_with_friend(self, persona: Persona, history: List[Message], user: Dict[str, Any]) -> str:
        messages = [
            {
                "role": "system",
                "content": self._build_system_prompt(persona, user),
            }
        ]
        messages.extend([msg.to_dict() for msg in history])

        if self.provider == "openai":
            return await self._openai_chat(messages)
        elif self.provider == "gemini":
            return await self._gemini_chat(messages)
        else:
            raise RuntimeError("Unsupported LLM provider: %s" % self.provider)

    def _build_system_prompt(self, persona: Persona, user: Dict[str, Any]) -> str:
        template = self._load_system_prompt_template()
        mapping = {
            "current_time": datetime.now(tz=timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S"),
            "persona": persona.prompt,
            "user_name": user.get("name", ""),
            "user_country": user.get("country", ""),
        }
        try:
            return template.format_map(mapping)
        except KeyError as exc:
            raise RuntimeError(
                f"System prompt template is missing field: {exc.args[0]}"
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

    async def _gemini_chat(self, messages: List[Dict[str, str]]) -> str:
        return await asyncio.to_thread(self._gemini_call, messages)

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

    def _gemini_call(self, messages: List[Dict[str, str]]) -> str:
        if not self.gemini:
            raise RuntimeError("Gemini client is not initialized.")

        params = self._load_llm_params()

        # build kwargs for the Gemini call, only include known keys
        allowed_keys = {
            "temperature",
            "maxOutputTokens",
            "topP",
            "presencePenalty",
            "frequencyPenalty",
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

        history = []
        system_prompt = ""

        # convert from openai format to Gemini format
        for msg in messages:
            if msg["role"] == "user":
                history.append(UserContent(parts=[Part(text=msg["content"])]))
            elif msg["role"] == "assistant":
                history.append(ModelContent(parts=[Part(text=msg["content"])]))
            elif msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                raise ValueError("Unknown message role: %s", msg["role"])

        message = messages[-1]["content"] if messages else ""
        history.pop()

        config = GenerateContentConfig(
            system_instruction=system_prompt,
            **call_kwargs,
        )

        chat = self.gemini.chats.create(
            model=params.get("model"),
            history=history,
            config=config,
        )
        response = chat.send_message(message)
        output = (response.text or "").strip()

        logger.info(
            "Gemini call model=%s usage=%s",
            getattr(response, "model_version", None),
            getattr(response, "usage_metadata", None),
        )

        return output
