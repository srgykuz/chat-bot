import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
import google.genai
import ollama
import jinja2
from pydantic import BaseModel, Field
import yaml

from src.config import get_logger, get_settings
from src.weather import WeatherInfo
from src.schema import (
    Message,
    MessageRole,
    Persona,
    User,
    Facts,
    EmotionalState,
    ConversationSummary,
    Relationships,
    Tool,
)
import src.limiter


logger = get_logger(__name__)
jinja = jinja2.Environment(
    autoescape=False,
    undefined=jinja2.StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


class ModelLimitReachedError(Exception):
    """
    Raised when a ModelClient usage has reached any of the limits defined in its config.
    """
    pass


@dataclass(slots=True)
class ModelResponse:
    """
    Result of LLM API call.
    """
    content: str
    usage_total_tokens: int


@dataclass(frozen=True, slots=True)
class ModelConfigLimits:
    """
    Rate limits for a ModelConfig.
    """
    rpd: int = 0
    tpd: int = 0

    def is_valid(self) -> bool:
        return bool(self.rpd >= 0 and self.tpd >= 0)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """
    Configuration of LLM API client.
    """
    provider: str
    model: str
    params: Dict[str, Any] = field(default_factory=dict)
    limits: ModelConfigLimits = field(default_factory=ModelConfigLimits)

    def is_valid(self) -> bool:
        return bool(self.provider and self.model and self.limits.is_valid())


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
        config: ModelConfig,
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
        config: ModelConfig,
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


class ModelClient:
    """
    Wrapper for interaction with LLM API of any provider.
    The provider and its parameters are loaded from the named configuration in "params.yml" file.
    """
    def __init__(self, name: str, use_limiter: bool = True) -> None:
        self.name = name
        self.use_limiter = use_limiter
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
        The provider is selected based on the loaded config.
        If no supported provider is configured, raises ValueError.
        """
        config = self.load_config(self.name)

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
        config = self.load_config(self.name)

        if not user_prompt:
            raise RuntimeError("User prompt is required.")

        if self.use_limiter and src.limiter.should_limit_llm(
            self.name,
            config.limits.rpd,
            config.limits.tpd,
        ):
            raise ModelLimitReachedError()

        response = await asyncio.to_thread(
            self.provider.generate,
            config,
            system_prompt,
            user_prompt,
            response_format,
        )
        response.content = self.format_assistant_response(response.content)

        if self.use_limiter:
            src.limiter.track_llm_rpd(self.name)
            src.limiter.track_llm_tpd(self.name, response.usage_total_tokens)

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
        config = self.load_config(self.name)

        if not conversation:
            raise RuntimeError("Conversation must contain at least one message.")

        if conversation[-1].role != MessageRole.USER:
            raise RuntimeError("The last message in the conversation must be from user.")

        if self.use_limiter and src.limiter.should_limit_llm(
            self.name,
            config.limits.rpd,
            config.limits.tpd,
        ):
            raise ModelLimitReachedError()

        context = [
            Message(role=MessageRole.SYSTEM, content=system_prompt)
        ] + conversation
        response = await asyncio.to_thread(
            self.provider.chat,
            config,
            context,
            response_format,
            tools,
        )
        response.content = self.format_assistant_response(response.content)

        if self.use_limiter:
            src.limiter.track_llm_rpd(self.name)
            src.limiter.track_llm_tpd(self.name, response.usage_total_tokens)

        return response

    def build_system_prompt(
        self,
        persona: Persona,
        user: User,
        persona_weather: Optional[WeatherInfo] = None,
        user_facts: Optional[Facts] = None,
        user_emotional_state: Optional[EmotionalState] = None,
        conversation_summary: Optional[ConversationSummary] = None,
        relationships: Optional[Relationships] = None,
        tools: Optional[List[Tool]] = None,
    ) -> str:
        """
        Creates a system prompt by loading the template and filling all the
        required placeholders. You should pass returned string as system prompt
        to the chat() method.
        """
        context = self.build_prompt_context(
            persona,
            user,
            persona_weather=persona_weather,
            user_facts=user_facts,
            user_emotional_state=user_emotional_state,
            conversation_summary=conversation_summary,
            relationships=relationships,
            tools=tools,
        )
        context["persona_prompt"] = self.build_persona_prompt(
            persona,
            user,
            persona_weather=persona_weather,
            user_facts=user_facts,
            user_emotional_state=user_emotional_state,
            conversation_summary=conversation_summary,
            relationships=relationships,
            tools=tools,
        )
        system_prompt = self.load_system_prompt()

        return jinja.from_string(system_prompt).render(context)

    def build_persona_prompt(
        self,
        persona: Persona,
        user: User,
        persona_weather: Optional[WeatherInfo] = None,
        user_facts: Optional[Facts] = None,
        user_emotional_state: Optional[EmotionalState] = None,
        conversation_summary: Optional[ConversationSummary] = None,
        relationships: Optional[Relationships] = None,
        tools: Optional[List[Tool]] = None,
    ) -> str:
        """
        Creates a persona-only prompt by rendering the persona template.
        """
        context = self.build_prompt_context(
            persona,
            user,
            persona_weather=persona_weather,
            user_facts=user_facts,
            user_emotional_state=user_emotional_state,
            conversation_summary=conversation_summary,
            relationships=relationships,
            tools=tools,
        )

        return jinja.from_string(persona.prompt).render(context)

    def build_prompt_context(
        self,
        persona: Persona,
        user: User,
        persona_weather: Optional[WeatherInfo] = None,
        user_facts: Optional[Facts] = None,
        user_emotional_state: Optional[EmotionalState] = None,
        conversation_summary: Optional[ConversationSummary] = None,
        relationships: Optional[Relationships] = None,
        tools: Optional[List[Tool]] = None,
    ) -> Dict[str, Any]:
        """
        Creates prompt context that can be used to enrich prompt template.
        """
        persona_dt = persona.now()
        persona_now = persona_dt.strftime("%Y-%m-%d %H:%M:%S")
        persona_weekday = persona_dt.weekday()

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
            "relationships": relationships,
            "tools": {t.name: t.description for t in tools} if tools else None,
        }

        return context

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

        config_limits = config.pop("limits", None)

        if config_limits:
            config["limits"] = ModelConfigLimits(**config_limits)

        model_config = ModelConfig(**config)

        if not model_config.is_valid():
            raise RuntimeError(f"Invalid model config: {name}")

        return model_config

    def format_assistant_response(self, content: str) -> str:
        """
        Formats assistant chat response to add more "humanity".
        """
        content = content.replace("—", "-")
        content = content.replace("«", '"')
        content = content.replace("»", '"')

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
        config: ModelConfig,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[type[BaseModel]] = None,
    ) -> ModelResponse:
        params = dict(config.params)

        params["model"] = config.model
        params["instructions"] = system_prompt
        params["input"] = user_prompt

        if response_format:
            params["text_format"] = response_format

        response = self.client.responses.parse(**params)
        result = ModelResponse(
            content=(response.output_text or ""),
            usage_total_tokens=0,
        )

        if response.usage:
            result.usage_total_tokens = response.usage.total_tokens

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
        config: ModelConfig,
        context: List[Message],
        response_format: Optional[type[BaseModel]] = None,
        tools: Optional[List[Tool]] = None,
    ) -> ModelResponse:
        messages: List[dict[str, Any]] = [
            {"role": msg.role.value, "content": msg.content}
            for msg in context
        ]

        params = dict(config.params)
        params["model"] = config.model
        params["messages"] = messages

        if response_format:
            params["response_format"] = response_format

        if tools:
            params["tools"] = [
                {"type": "function", "function": t.definition(strict=True)}
                for t in tools
            ]

        response = None
        count = 0
        usage_total_tokens = 0

        while True:
            if count >= 10:
                raise RuntimeError("Infinite loop protection.")

            response = self.client.chat.completions.parse(**params)
            count += 1
            message = response.choices[0].message

            if response.usage:
                usage_total_tokens += response.usage.total_tokens

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
            content=(response.choices[0].message.content or ""),
            usage_total_tokens=usage_total_tokens,
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

        self.client = google.genai.Client(api_key=self.parent.settings.google_api_key)

    def close(self) -> None:
        self.client.close()

    def generate(
        self,
        config: ModelConfig,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[type[BaseModel]] = None,
    ) -> ModelResponse:
        params = dict(config.params)

        if response_format:
            params["responseMimeType"] = "application/json"
            params["responseJsonSchema"] = response_format.model_json_schema()

        generate_config = google.genai.types.GenerateContentConfig(
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
            usage_total_tokens=0,
        )

        if response.usage_metadata:
            result.usage_total_tokens = response.usage_metadata.total_token_count or 0

        logger.info(
            "Google generate: model=%s params=%s usage=%s",
            getattr(response, "model_version", None),
            params,
            getattr(response, "usage_metadata", None),
        )

        return result

    def chat(
        self,
        config: ModelConfig,
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
                content = google.genai.types.UserContent(parts=[google.genai.types.Part(text=msg.content)])
            elif msg.role == MessageRole.ASSISTANT:
                content = google.genai.types.ModelContent(parts=[google.genai.types.Part(text=msg.content)])
            else:
                raise ValueError(f"Unknown message role: {msg.role}")

            if history and isinstance(content, type(history[-1])):
                history[-1].parts[0].text += f"\n{content.parts[0].text}"
            else:
                history.append(content)

        if not history:
            raise ValueError("History cannot be empty.")

        last = history.pop()

        if not isinstance(last, google.genai.types.UserContent):
            raise ValueError("Last message in context should be from user")

        curr_message = last.parts[0].text

        if response_format and tools and ("gemini-2" in config.model):
            raise RuntimeError("Function calling with Structured output is available only for Gemini 3 models.")

        params = dict(config.params)

        if response_format:
            params["responseMimeType"] = "application/json"
            params["responseJsonSchema"] = response_format.model_json_schema()

        if tools:
            params["tools"] = [t.f for t in tools]

        generate_config = google.genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            **params,
        )
        chat = self.client.chats.create(
            model=config.model,
            history=history,
            config=generate_config,
        )

        response = chat.send_message(curr_message)
        result = ModelResponse(
            content=(response.text or ""),
            usage_total_tokens=0,
        )

        if response.usage_metadata:
            result.usage_total_tokens = response.usage_metadata.total_token_count or 0

        logger.info(
            "Google chat: model=%s params=%s usage=%s",
            getattr(response, "model_version", None),
            params,
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
        config: ModelConfig,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[type[BaseModel]] = None,
    ) -> ModelResponse:
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
            usage_total_tokens=((response.prompt_eval_count or 0) + (response.eval_count or 0)),
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
        config: ModelConfig,
        context: List[Message],
        response_format: Optional[type[BaseModel]] = None,
        tools: Optional[List[Tool]] = None,
    ) -> ModelResponse:
        messages = [
            ollama.Message(role=msg.role.value, content=msg.content)
            for msg in context
        ]
        params = dict(config.params)

        if response_format:
            params["format"] = response_format.model_json_schema()

        if tools:
            params["tools"] = [t.f for t in tools]

        response: Optional[ollama.ChatResponse] = None
        count = 0

        while True:
            if count >= 10:
                raise RuntimeError("Infinite loop protection.")

            response = self.client.chat(model=config.model, messages=messages, **params)
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
            usage_total_tokens=((response.prompt_eval_count or 0) + (response.eval_count or 0)),
        )

        params_log = dict(params)
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


def history_to_conversation(history: list[Message]) -> str:
    """
    Converts chat history into format suitable for passing into LLM as a prompt.
    """
    return "\n".join([f"{msg.role.value.title()}: {msg.content}" for msg in history])


if __name__ == "__main__":
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

    client = ModelClient("chat", use_limiter=False)
    response = asyncio.run(
        client.chat(
            system_prompt,
            conversation,
            response_format=CalendarEvent,
            tools=tools,
        )
    )
    result = CalendarEvent.model_validate_json(response.content)

    print(response)
    print(result.model_dump_json(indent=2))
