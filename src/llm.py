import logging
import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
from google import genai
from google.genai import types as genai_types
import ollama
from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel, Field
import yaml

from src.config import get_settings
from src.weather import WeatherInfo
from src.schema import Message, MessageRole, Persona, User, Facts, EmotionalState, ConversationSummary, Tool


logger = logging.getLogger(__name__)
jinja = Environment(
    autoescape=False,
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


@dataclass(slots=True)
class ModelResponse:
    """
    Result of LLM API call.
    """
    content: str


class ProviderClient(ABC):
    """
    Base class that should be implemented by provider-specific LLM client.
    """
    def __init__(self, parent: "ModelClient") -> None:
        self.parent = parent

    @abstractmethod
    def close(self) -> None:
        """
        Closes an underlying resources.
        """
        pass

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[type[BaseModel]] = None,
    ) -> ModelResponse:
        """
        Generates a response for a single-turn prompt.

        The prompt consist of a system instruction and a single user message
        the model should respond to.

        Response is a generated assistant message text. If response_format is provided,
        content is a JSON string.
        """
        pass

    @abstractmethod
    def chat(
        self,
        context: List[Message],
        response_format: Optional[type[BaseModel]] = None,
        tools: Optional[List[Tool]] = None,
    ) -> ModelResponse:
        """
        Generates a response for the supplied chat context.

        The context consist of system prompt, past user and assistant messages,
        and user's current message the model should respond to.

        Response is a generated assistant message text. If response_format is provided,
        content is a JSON string.
        """
        pass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """
    Configuration of LLM API provider.
    """
    provider: str
    model: str
    params: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        return bool(self.provider and self.model)


class ModelClient:
    """
    Wrapper for interaction with LLM API of any provider.
    The provider and its parameters are loaded from the named configuration in "params.yml" file.
    """
    def __init__(self, config_name: str) -> None:
        self.settings = get_settings()
        self.config_name = config_name
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
        The provider is selected based on the loaded config.
        If no supported provider is configured, raises ValueError.
        """
        config = self.load_config(self.config_name)

        if config.provider == "openai":
            return OpenAIClient(self)

        if config.provider == "google":
            return GoogleClient(self)

        if config.provider == "ollama":
            return OllamaClient(self)

        raise ValueError(f"Unsupported provider: {config.provider}")

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[type[BaseModel]] = None,
    ) -> ModelResponse:
        """
        Takes pre-built prompts, calls LLM API and returns generated response.

        system_prompt is a system instructions, user_prompt is a user request the
        model should respond to.

        Returns model response. If response_format is provided, content is a JSON string
        which you should parse and validate using Pydantic's model_validate_json().
        """
        if not user_prompt:
            raise RuntimeError("User prompt is required.")

        response = await asyncio.to_thread(
            self.provider.generate,
            system_prompt,
            user_prompt,
            response_format,
        )

        return response

    async def chat(
        self,
        system_prompt: str,
        conversation: List[Message],
        response_format: Optional[type[BaseModel]] = None,
        tools: Optional[List[Tool]] = None,
    ) -> ModelResponse:
        """
        Builds full chat context, calls LLM API and returns generated response
        to the last user messages.

        system_prompt should be created using build_system_prompt(). Create it
        for every new chat() call.

        conversation should contain all previous messages from both user and assistant,
        and should contain user's current message the model should respond to. Sorted from
        oldest to newest.

        Returns model response. If response_format is provided, content is a JSON string
        which you should parse and validate using Pydantic's model_validate_json().
        """
        if not conversation:
            raise RuntimeError("Conversation must contain at least one message.")

        if conversation[-1].role != MessageRole.USER:
            raise RuntimeError("The last message in the conversation must be from user.")

        context = [
            Message(role=MessageRole.SYSTEM, content=system_prompt)
        ] + conversation
        response = await asyncio.to_thread(self.provider.chat, context, response_format, tools)
        response.content = self.format_assistant_response(response.content)

        return response

    def build_system_prompt(
        self,
        persona: Persona,
        user: User,
        persona_weather: Optional[WeatherInfo] = None,
        user_facts: Optional[Facts] = None,
        user_emotional_state: Optional[EmotionalState] = None,
        conversation_summary: Optional[ConversationSummary] = None,
    ) -> str:
        """
        Creates a system prompt by loading the template and filling all the
        required placeholders. You should pass returned string as system prompt
        to the chat() method.
        """
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

        context = {
            "settings": self.settings,
            "persona": persona,
            "user": user,
            "user_facts": user_facts,
            "user_emotional_state": user_emotional_state,
            "persona_now": persona_now,
            "persona_weekday": persona_weekday,
            "persona_weather": persona_weather,
            "conversation_summary": conversation_summary,
        }

        persona_prompt = jinja.from_string(persona.prompt).render(context)
        context["persona_prompt"] = persona_prompt

        system_prompt = self.load_system_prompt()

        return jinja.from_string(system_prompt).render(context)

    def load_system_prompt(self) -> str:
        """
        Loads the system prompt from "prompt.md" file.
        """
        path = Path(self.settings.system_path) / "prompt.md"

        if not path.exists():
            raise RuntimeError(f"System prompt file not found: {path}")

        return path.read_text(encoding="utf-8")

    def load_config(self, name: str) -> ModelConfig:
        """
        Loads the named chat model configuration from "params.yml" file.
        """
        path = Path(self.settings.system_path) / "params.yml"

        if not path.exists():
            raise RuntimeError(f"Params file not found: {path}")

        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)

        if not isinstance(data, dict):
            raise RuntimeError("Invalid params file format.")

        config = data.get(name)

        if not isinstance(config, dict):
            raise RuntimeError(f"Config not found: {name}")

        model_config = ModelConfig(**config)

        if not model_config.is_valid():
            raise RuntimeError(f"Invalid model config: {name}")

        return model_config

    def format_assistant_response(self, content: str) -> str:
        """
        Formats assistant chat response to add more "humanity".
        """
        content = content.replace("—", "-")

        return content


class OpenAIClient(ProviderClient):
    def __init__(self, parent: "ModelClient") -> None:
        super().__init__(parent)

        if not self.parent.settings.openai_api_key:
            raise RuntimeError("OpenAI API key is not configured.")

        self.client = openai.OpenAI(api_key=self.parent.settings.openai_api_key)

    def close(self) -> None:
        self.client.close()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[type[BaseModel]] = None,
    ) -> ModelResponse:
        config = self.parent.load_config(self.parent.config_name)
        params = dict(config.params)

        params["model"] = config.model
        params["instructions"] = system_prompt
        params["input"] = user_prompt

        if response_format:
            params["text_format"] = response_format

        response = self.client.responses.parse(**params)
        result = ModelResponse(
            content=(response.output_text or "")
        )

        params_log = dict(params)
        params_log.pop("input", None)
        logger.info(
            "OpenAI generate: model=%s params=%s usage=%s",
            getattr(response, "model", None),
            params_log,
            getattr(response, "usage", None),
        )

        return result

    def chat(
        self,
        context: List[Message],
        response_format: Optional[type[BaseModel]] = None,
        tools: Optional[List[Tool]] = None,
    ) -> ModelResponse:
        config = self.parent.load_config(self.parent.config_name)

        messages: List[dict[str, Any]] = [
            {"role": msg.role.value, "content": msg.content}
            for msg in context
        ]
        config.params["messages"] = messages
        config.params["model"] = config.model

        if response_format:
            config.params["response_format"] = response_format

        if tools:
            config.params["tools"] = [
                {"type": "function", "function": t.definition(strict=True)}
                for t in tools
            ]

        response = None
        count = 0

        while True:
            if count >= 10:
                raise RuntimeError("Infinite loop protection.")

            response = self.client.chat.completions.parse(**config.params)
            count += 1
            message = response.choices[0].message

            if message.tool_calls:
                if not tools:
                    raise RuntimeError("No tools defined.")

                messages.append({
                    "role": MessageRole.ASSISTANT.value,
                    "tool_calls": message.tool_calls,
                })

                for call in message.tool_calls:
                    for tool in tools:
                        if call.function.name == tool.name:
                            args = json.loads(call.function.arguments)
                            result = tool.f(**args)

                            messages.append({
                                "role": MessageRole.TOOL.value,
                                "content": str(result),
                                "tool_call_id": call.id,
                            })

                            break
            else:
                break

        result = ModelResponse(
            content=(response.choices[0].message.content or "")
        )
        params_log = dict(config.params)

        params_log.pop("messages", None)
        params_log.pop("response_format", None)
        params_log.pop("tools", None)

        logger.info(
            "OpenAI chat: model=%s params=%s usage=%s",
            getattr(response, "model", None),
            params_log,
            getattr(response, "usage", None),
        )

        return result


class GoogleClient(ProviderClient):
    def __init__(self, parent: "ModelClient") -> None:
        super().__init__(parent)

        if not self.parent.settings.google_api_key:
            raise RuntimeError("Google API key is not configured.")

        self.client = genai.Client(api_key=self.parent.settings.google_api_key)

    def close(self) -> None:
        self.client.close()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[type[BaseModel]] = None,
    ) -> ModelResponse:
        config = self.parent.load_config(self.parent.config_name)
        params = dict(config.params)

        if response_format:
            params["responseMimeType"] = "application/json"
            params["responseJsonSchema"] = response_format.model_json_schema()

        generate_config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            **params,
        )
        response = self.client.models.generate_content(
            model=config.model,
            contents=user_prompt,
            config=generate_config,
        )
        result = ModelResponse(
            content=(response.text or ""),
        )

        logger.info(
            "Google generate: model=%s params=%s usage=%s",
            getattr(response, "model_version", None),
            params,
            getattr(response, "usage_metadata", None),
        )

        return result

    def chat(
        self,
        context: List[Message],
        response_format: Optional[type[BaseModel]] = None,
        tools: Optional[List[Tool]] = None,
    ) -> ModelResponse:
        system_prompt = ""
        history = []

        for msg in context:
            content = None

            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
                continue
            elif msg.role == MessageRole.USER:
                content = genai_types.UserContent(parts=[genai_types.Part(text=msg.content)])
            elif msg.role == MessageRole.ASSISTANT:
                content = genai_types.ModelContent(parts=[genai_types.Part(text=msg.content)])
            else:
                raise ValueError(f"Unknown message role: {msg.role}")

            if history and isinstance(content, type(history[-1])):
                history[-1].parts[0].text += f"\n{content.parts[0].text}"
            else:
                history.append(content)

        if not history:
            raise ValueError("History cannot be empty.")

        last = history.pop()

        if not isinstance(last, genai_types.UserContent):
            raise ValueError("Last message in context should be from user")

        curr_message = last.parts[0].text
        config = self.parent.load_config(self.parent.config_name)

        if response_format and tools and ("gemini-2" in config.model):
            raise RuntimeError("Function calling with Structured output is available only for Gemini 3 models.")

        if response_format:
            config.params["responseMimeType"] = "application/json"
            config.params["responseJsonSchema"] = response_format.model_json_schema()

        if tools:
            config.params["tools"] = [t.f for t in tools]

        generate_config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            **config.params,
        )
        chat = self.client.chats.create(
            model=config.model,
            history=history,
            config=generate_config,
        )

        response = chat.send_message(curr_message)
        result = ModelResponse(
            content=(response.text or "")
        )

        logger.info(
            "Google chat: model=%s params=%s usage=%s",
            getattr(response, "model_version", None),
            config.params,
            getattr(response, "usage_metadata", None),
        )

        return result


class OllamaClient(ProviderClient):
    def __init__(self, parent: "ModelClient") -> None:
        super().__init__(parent)

        if not self.parent.settings.ollama_host:
            raise RuntimeError("Ollama host is not configured.")

        headers = {}

        if self.parent.settings.ollama_api_key:
            headers["Authorization"] = f"Bearer {self.parent.settings.ollama_api_key}"

        self.client = ollama.Client(host=self.parent.settings.ollama_host, headers=headers)

    def close(self) -> None:
        self.client.close()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[type[BaseModel]] = None,
    ) -> ModelResponse:
        config = self.parent.load_config(self.parent.config_name)
        params = dict(config.params)

        if response_format:
            params["format"] = response_format.model_json_schema()

        response = self.client.generate(
            model=config.model,
            prompt=user_prompt,
            system=system_prompt,
            **params,
        )
        result = ModelResponse(
            content=response.response,
        )

        usage = {
            "total_duration": response.total_duration / 1e9,
            "load_duration": response.load_duration / 1e9,
            "prompt_eval_count": response.prompt_eval_count,
            "prompt_eval_duration": response.prompt_eval_duration / 1e9,
            "eval_count": response.eval_count,
            "eval_duration": response.eval_duration / 1e9,
        }

        logger.info(
            "Ollama generate: model=%s params=%s usage=%s",
            config.model,
            params,
            usage,
        )

        return result

    def chat(
        self,
        context: List[Message],
        response_format: Optional[type[BaseModel]] = None,
        tools: Optional[List[Tool]] = None,
    ) -> ModelResponse:
        config = self.parent.load_config(self.parent.config_name)
        messages = [
            ollama.Message(role=msg.role.value, content=msg.content)
            for msg in context
        ]

        if response_format:
            config.params["format"] = response_format.model_json_schema()

        if tools:
            config.params["tools"] = [t.f for t in tools]

        response: Optional[ollama.ChatResponse] = None
        count = 0

        while True:
            if count >= 10:
                raise RuntimeError("Infinite loop protection.")

            response = self.client.chat(model=config.model, messages=messages, **config.params)
            count += 1

            if not response:
                raise RuntimeError("No response.")

            if response.message.tool_calls:
                if not tools:
                    raise RuntimeError("No tools defined.")

                messages.append(response.message)

                for call in response.message.tool_calls:
                    for tool in tools:
                        if call.function.name == tool.name:
                            result = tool.f(**call.function.arguments)

                            messages.append(ollama.Message(
                                role=MessageRole.TOOL.value,
                                content=str(result),
                                tool_name=call.function.name,
                            ))

                            break
            else:
                break

        result = ModelResponse(
            content=(response.message.content or ""),
        )

        params_log = dict(config.params)
        params_log.pop("format", None)
        params_log.pop("tools", None)

        usage = {
            "total_duration": (response.total_duration or 0) / 1e9,
            "load_duration": (response.load_duration or 0) / 1e9,
            "prompt_eval_count": (response.prompt_eval_count or 0),
            "prompt_eval_duration": (response.prompt_eval_duration or 0) / 1e9,
            "eval_count": (response.eval_count or 0),
            "eval_duration": (response.eval_duration or 0) / 1e9,
        }

        logger.info(
            "Ollama chat: model=%s params=%s usage=%s",
            config.model,
            params_log,
            usage,
        )

        return result


if __name__ == "__main__":
    import asyncio

    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    class GetParticipantsParams(BaseModel):
        day: str = Field(description="Day of the week.")

    def get_participants(day: str) -> str:
        """Returns a participants of the event according to the weekday.

        Args:
            day: Day of the week.

        Returns:
            A participants.
        """
        if day.lower() == "friday":
            return "Alice and Bob"

        return ""

    system_prompt = (
        "Extract the event information. "
        "Call function to get the event participants. "
        "Return result as a JSON string strictly matching the requested schema."
    )
    conversation = [
        Message(
            role=MessageRole.USER,
            content="They are going to a science fair on Friday."
        )
    ]
    tools = [
        Tool(
            f=get_participants,
            params=GetParticipantsParams,
        )
    ]

    client = ModelClient("chat")
    response = asyncio.run(
        client.chat(
            system_prompt,
            conversation,
            response_format=CalendarEvent,
            tools=tools,
        )
    )
    result = CalendarEvent.model_validate_json(response.content)

    print(result)
