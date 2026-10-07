"""Parse structured provider responses into Pydantic models."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel


ModelT = TypeVar("ModelT", bound=BaseModel)
def parse_structured_response(response: Any, model: type[ModelT]) -> ModelT:
    """Return a Pydantic model from Gemini's parsed or JSON response."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, model):
        return parsed
    if isinstance(parsed, dict):
        return model.model_validate(parsed)
    if isinstance(parsed, str):
        return model.model_validate_json(parsed)

    response_text = getattr(response, "text", None)
    if not response_text:
        choices = getattr(response, "choices", None)
        if isinstance(choices, (list, tuple)) and choices:
            response_text = getattr(getattr(choices[0], "message", None), "content", None)
    if isinstance(response_text, str) and response_text:
        return model.model_validate_json(response_text)

    raise ValueError("Gemini did not produce a parseable structured response.")
