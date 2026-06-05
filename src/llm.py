import json
import logging
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, List

import openai
from google import genai
from google.genai.types import GenerateContentConfig, ModelContent, Part, UserContent

from src.config import get_settings
from src.session import Message, MessageRole, Persona, User


logger = logging.getLogger(__name__)


class ProviderClient(ABC):
    """
    Base class that should be implemented by provider-specific LLM client.
    """
    name = ""

    def __init__(self, parent: "ModelClient") -> None:
        self.parent = parent

    @abstractmethod
    def close(self) -> None:
        """
        Closes an underlying resources.
        """
        pass

    @abstractmethod
    def chat(self, context: List[Message]) -> str:
        """
        Generates a response for the supplied chat context.
        The context consist of system prompt, past user and assistant messages,
        and user's current message the model should respond to.
        Output is a generated assistant message text.
        """
        pass


class ModelClient:
    """
    Wrapper for interaction with LLM API of any provider.
    The provider is selected based on the settings (e.g. OpenAI, Gemini).
    """
    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider: ProviderClient = self.create_provider()

    def close(self) -> None:
        """
        Closes an underlying resources.
        """
        self.provider.close()

    def create_provider(self) -> ProviderClient:
        """
        Creates and returns an instance of the provider client.
        Using this instance you can interact with specific LLM API.
        The provider is selected based on the settings.
        If no supported provider is configured, raises RuntimeError.
        """
        if self.settings.openai_api_key:
            return OpenAIClient(self)

        if self.settings.gemini_api_key:
            return GeminiClient(self)

        raise RuntimeError("No supported LLM provider is configured.")

    async def chat(self, system_prompt: str, conversation: List[Message]) -> str:
        """
        Builds full chat context, calls LLM API and returns generated response.

        system_prompt should be created using build_system_prompt(). Create it
        for every new chat() call.

        conversation should contain all previous messages from both user and assistant,
        and should contain user's current message the model should respond to. Sorted from
        oldest to newest.
        """
        if not conversation:
            raise RuntimeError("Conversation must contain at least one message.")

        if conversation[-1].role != MessageRole.USER:
            raise RuntimeError("The last message in the conversation must be from user.")

        context = [
            Message(role=MessageRole.SYSTEM, content=system_prompt)
        ] + conversation

        output = await asyncio.to_thread(self.provider.chat, context)
        output = output.strip()

        return output

    def build_system_prompt(self, persona: Persona, user: User) -> str:
        """
        Creates a system prompt by loading the template and filling all the
        required placeholders. You should pass returned string as system prompt
        to the chat() method.
        """
        template = self.load_system_prompt()

        persona_tz = ZoneInfo(persona.timezone)
        persona_dt = datetime.now(tz=persona_tz)
        persona_now = persona_dt.strftime("%Y-%m-%d %H:%M:%S")
        persona_weekday = [
            "Понедельник",
            "Вторник",
            "Среда",
            "Четверг",
            "Пятница",
            "Суббота",
            "Воскресенье",
        ][persona_dt.weekday()]
        persona_time = f"{persona_now} {persona_weekday}"

        mapping = {
            "persona_time": persona_time,
            "persona_prompt": persona.prompt,
            "user_name": user.first_name or "",
            "user_country": user.country() or "",
        }

        return template.format_map(mapping)

    def load_system_prompt(self) -> str:
        """
        Loads the system prompt from the configured text file path.
        """
        path = Path(self.settings.system_prompt_path)

        if not path.exists():
            raise RuntimeError(f"System prompt file not found: {path}")

        return path.read_text(encoding="utf-8")

    def load_model_params(self) -> Dict[str, Any]:
        """
        Loads the model parameters from the configured JSON file path.
        """
        path = Path(self.settings.model_params_path)

        if not path.exists():
            raise RuntimeError(f"Model params file not found: {path}")

        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)

        if not isinstance(data, dict):
            raise RuntimeError("Model params file must contain a JSON object at the top level.")

        return data


class OpenAIClient(ProviderClient):
    name = "openai"

    def __init__(self, parent: "ModelClient") -> None:
        super().__init__(parent)
        self.client = openai.OpenAI(api_key=self.parent.settings.openai_api_key)

    def close(self) -> None:
        self.client.close()

    def chat(self, context: List[Message]) -> str:
        params = self.parent.load_model_params()

        messages = [msg.to_dict() for msg in context]
        params["messages"] = messages

        completion = self.client.chat.completions.create(**params)

        if not completion.choices:
            raise RuntimeError("No response.")

        output = completion.choices[0].message.content or ""

        params_log = dict(params)
        params_log.pop("messages", None)
        logger.info(
            "OpenAI call model=%s params=%s usage=%s",
            getattr(completion, "model", None),
            params_log,
            getattr(completion, "usage", None),
        )

        return output


class GeminiClient(ProviderClient):
    name = "gemini"

    def __init__(self, parent: "ModelClient") -> None:
        super().__init__(parent)
        self.client = genai.Client(api_key=self.parent.settings.gemini_api_key)

    def close(self) -> None:
        self.client.close()

    def chat(self, context: List[Message]) -> str:
        system_prompt = ""
        curr_message = context[-1].content
        history = []

        for msg in context[:-1]:
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
            elif msg.role == MessageRole.USER:
                history.append(UserContent(parts=[Part(text=msg.content)]))
            elif msg.role == MessageRole.ASSISTANT:
                history.append(ModelContent(parts=[Part(text=msg.content)]))
            else:
                raise ValueError(f"Unknown message role: {msg.role}")

        params = self.parent.load_model_params()
        model = params.pop("model", "")

        config = GenerateContentConfig(
            system_instruction=system_prompt,
            **params,
        )
        chat = self.client.chats.create(
            model=model,
            history=history,
            config=config,
        )

        response = chat.send_message(curr_message)
        output = response.text or ""

        logger.info(
            "Gemini call model=%s params=%s usage=%s",
            getattr(response, "model_version", None),
            params,
            getattr(response, "usage_metadata", None),
        )

        return output
