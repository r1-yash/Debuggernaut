"""Parse structured Gemini responses into Pydantic models."""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from pydantic import BaseModel


ModelT = TypeVar("ModelT", bound=BaseModel)
_LOGGER = logging.getLogger(__name__)
_gemini_call_count = 0


#logging, max calls =4 

def log_gemini_call(label: str, response: Any) -> None:
    """Log Gemini token usage and the running number of completed calls."""
    global _gemini_call_count
    _gemini_call_count += 1

    usage_metadata = getattr(response, "usage_metadata", None)
    _LOGGER.info(
        "%s prompt_tokens=%s candidates_tokens=%s total_tokens=%s "
        "gemini_call_count=%d",
        label,
        getattr(usage_metadata, "prompt_token_count", None),
        getattr(usage_metadata, "candidates_token_count", None),
        getattr(usage_metadata, "total_token_count", None),
        _gemini_call_count,
    )

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
    if isinstance(response_text, str) and response_text:
        return model.model_validate_json(response_text)

    raise ValueError("Gemini did not produce a parseable structured response.")
