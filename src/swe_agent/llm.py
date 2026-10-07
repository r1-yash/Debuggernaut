"""Provider-neutral LLM clients and per-run usage accounting."""

from __future__ import annotations

import contextvars
import inspect
import json
import os
import time
import types as stdlib_types
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints

from google.genai import types
from pydantic import BaseModel


QWEN_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_QWEN_MODEL = "qwen/qwen3-coder:free"


def _annotation_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Convert common Python type annotations to JSON Schema for Qwen tools."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is list:
        return {"type": "array", "items": {}}
    if annotation is dict:
        return {"type": "object", "additionalProperties": {}}
    if annotation is type(None):
        return {"type": "null"}

    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is list:
        return {
            "type": "array",
            "items": _annotation_to_json_schema(arguments[0]) if arguments else {},
        }
    if origin is dict:
        return {
            "type": "object",
            "additionalProperties": (
                _annotation_to_json_schema(arguments[1]) if len(arguments) > 1 else {}
            ),
        }
    if origin in (Union, stdlib_types.UnionType):
        return {"anyOf": [_annotation_to_json_schema(argument) for argument in arguments]}
    return {}


def _qwen_tool_schema(tool: Callable[..., Any]) -> dict[str, Any]:
    """Return an OpenAI-compatible function-tool schema for a Python callable."""
    signature = inspect.signature(tool)
    try:
        annotations = get_type_hints(tool)
    except (NameError, TypeError):
        annotations = {}
    parameters_by_name = {
        name: parameter
        for name, parameter in signature.parameters.items()
        if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
    }
    properties = {
        name: _annotation_to_json_schema(annotations.get(name, parameter.annotation))
        for name, parameter in parameters_by_name.items()
    }
    required = [
        name for name, parameter in parameters_by_name.items()
        if parameter.default is parameter.empty
    ]
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        parameters["required"] = required
    return {
        "type": "function",
        "function": {
            "name": tool.__name__,
            "description": tool.__doc__ or "",
            "parameters": parameters,
        },
    }


class LLMUsage(BaseModel):
    """Token usage returned by a provider, or ``None`` when unavailable."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class LLMCall(BaseModel):
    """One requested provider operation in a Debuggernaut run."""

    provider: str
    model: str
    attempt: int | None
    call_number: int
    purpose: str
    timestamp: datetime
    success: bool
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_seconds: float | None = None


class UsageTracker:
    """Owns call numbering and aggregates provider-normalized usage."""

    def __init__(self) -> None:
        self.calls: list[LLMCall] = []
        self.attempt: int | None = None

    def set_attempt(self, attempt: int) -> None:
        self.attempt = attempt

    def record(
        self, provider: str, model: str, purpose: str, response: Any = None,
        *, success: bool, latency_seconds: float | None = None,
    ) -> None:
        usage = normalize_usage(response)
        self.calls.append(LLMCall(
            provider=provider, model=model, attempt=self.attempt,
            call_number=len(self.calls) + 1, purpose=purpose,
            timestamp=datetime.now(UTC), success=success,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens, latency_seconds=latency_seconds,
        ))

    def totals(self) -> LLMUsage:
        def total(field: str) -> int | None:
            values = [getattr(call, field) for call in self.calls]
            return sum(values) if values and all(value is not None for value in values) else None
        return LLMUsage(input_tokens=total("input_tokens"), output_tokens=total("output_tokens"), total_tokens=total("total_tokens"))


def normalize_usage(response: Any) -> LLMUsage:
    """Read supported provider usage fields without estimating absent values."""
    usage = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
    if usage is None:
        return LLMUsage()
    def value(*names: str) -> int | None:
        for name in names:
            candidate = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                return candidate
        return None
    return LLMUsage(
        input_tokens=value("prompt_token_count", "input_tokens", "prompt_tokens"),
        output_tokens=value("candidates_token_count", "output_tokens", "completion_tokens"),
        total_tokens=value("total_token_count", "total_tokens"),
    )


_ACTIVE_TRACKER: contextvars.ContextVar[UsageTracker | None] = contextvars.ContextVar("usage_tracker", default=None)


class usage_run:
    """Context manager that makes one tracker available to wrapped raw clients."""
    def __init__(self, tracker: UsageTracker) -> None:
        self.tracker = tracker
        self.token: contextvars.Token[UsageTracker | None] | None = None
    def __enter__(self) -> UsageTracker:
        self.token = _ACTIVE_TRACKER.set(self.tracker)
        return self.tracker
    def __exit__(self, *args: Any) -> None:
        assert self.token is not None
        _ACTIVE_TRACKER.reset(self.token)


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str

    @classmethod
    def from_environment(cls) -> "LLMSettings":
        provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        if provider not in {"gemini", "qwen"}:
            raise ValueError("LLM_PROVIDER must be either 'gemini' or 'qwen'.")
        default = DEFAULT_QWEN_MODEL if provider == "qwen" else DEFAULT_GEMINI_MODEL
        return cls(provider=provider, model=os.getenv("LLM_MODEL", default).strip() or default)


class LLMProvider:
    provider: str
    model: str
    def explore(self, system_instruction: str, prompt: str, tools: list[Callable[..., Any]]) -> Any: ...
    def structured(self, system_instruction: str, contents: Any, schema: type[BaseModel], purpose: str) -> Any: ...


class GeminiProvider(LLMProvider):
    provider = "gemini"
    def __init__(self, client: Any, model: str = DEFAULT_GEMINI_MODEL) -> None:
        self.client, self.model = client, model
    def _call(self, purpose: str, operation: Callable[[], Any]) -> Any:
        started = time.monotonic()
        try:
            response = operation()
        except Exception:
            _record(self.provider, self.model, purpose, success=False, latency_seconds=time.monotonic() - started)
            raise
        _record(self.provider, self.model, purpose, response, success=True, latency_seconds=time.monotonic() - started)
        return response
    def explore(self, system_instruction: str, prompt: str, tools: list[Callable[..., Any]]) -> Any:
        chat = self.client.chats.create(model=self.model, config=types.GenerateContentConfig(
            system_instruction=system_instruction, tools=tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=4),
        ))
        response = self._call("exploration", lambda: chat.send_message(prompt))
        return chat, response
    def structured(self, system_instruction: str, contents: Any, schema: type[BaseModel], purpose: str) -> Any:
        return self._call(purpose, lambda: self.client.models.generate_content(
            model=self.model, contents=contents,
            config=types.GenerateContentConfig(system_instruction=system_instruction, response_mime_type="application/json", response_schema=schema),
        ))


class QwenProvider(LLMProvider):
    provider = "qwen"
    def __init__(self, model: str, api_key: str, base_url: str = QWEN_OPENROUTER_BASE_URL) -> None:
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    def _call(self, purpose: str, **kwargs: Any) -> Any:
        started = time.monotonic()
        try:
            response = self.client.chat.completions.create(model=self.model, **kwargs)
        except Exception:
            _record(self.provider, self.model, purpose, success=False, latency_seconds=time.monotonic() - started)
            raise
        _record(self.provider, self.model, purpose, response, success=True, latency_seconds=time.monotonic() - started)
        return response
    def explore(self, system_instruction: str, prompt: str, tools: list[Callable[..., Any]]) -> Any:
        functions = {tool.__name__: tool for tool in tools}
        tool_specs = [_qwen_tool_schema(tool) for tool in tools]
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_instruction}, {"role": "user", "content": prompt}]
        for _ in range(4):
            response = self._call("exploration", messages=messages, tools=tool_specs)
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            if not message.tool_calls:
                return messages, response
            for call in message.tool_calls:
                try:
                    result = functions[call.function.name](**json.loads(call.function.arguments or "{}"))
                    content = json.dumps(result)
                except Exception as error:
                    content = json.dumps({"error": str(error)})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
        raise ValueError("Qwen exhausted the repository tool-call limit without a final response.")
    def structured(self, system_instruction: str, contents: Any, schema: type[BaseModel], purpose: str) -> Any:
        messages = list(contents) if isinstance(contents, list) else [{"role": "user", "content": str(contents)}]
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": system_instruction})
        return self._call(purpose, messages=messages, response_format={"type": "json_object"})


def _record(provider: str, model: str, purpose: str, response: Any = None, *, success: bool, latency_seconds: float) -> None:
    tracker = _ACTIVE_TRACKER.get()
    if tracker is not None:
        tracker.record(provider, model, purpose, response, success=success, latency_seconds=latency_seconds)


def coerce_provider(client: Any) -> LLMProvider:
    """Adapt the old Gemini SDK client or return an already-configured provider."""
    if isinstance(client, LLMProvider):
        return client
    return GeminiProvider(client)


def provider_from_environment() -> LLMProvider:
    settings = LLMSettings.from_environment()
    if settings.provider == "gemini":
        from google import genai
        return GeminiProvider(genai.Client(), settings.model)
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        raise ValueError("QWEN_API_KEY must be set when LLM_PROVIDER=qwen.")
    return QwenProvider(settings.model, api_key, os.getenv("QWEN_BASE_URL", QWEN_OPENROUTER_BASE_URL))
