"""Preflight input-token counting for DeepSeek V4.1 chat requests.

The counter renders and tokenizes the *complete* outgoing request with the
official DeepSeek prompt encoding, so the configured input limit is checked
against the real prompt instead of a byte-length guess:

* ``deepseek-recipe`` (the maintained official toolkit, MIT) converts the
  OpenAI-compatible request body into a DeepSeek V4.1 conversation and renders
  it with the same prompt template the model is served with, including message
  boundaries, control tokens, tool definitions, tool calls, tool results,
  thinking-mode formatting, and the system prompt.
* ``tokenizer.json`` from ``deepseek-ai/DeepSeek-V4.1-Flash`` (about 6 MB; the
  tokenizer only, never model weights) turns that rendered prompt into token
  IDs.

Runtime resources
-----------------

``deepseek-recipe`` is declared in ``pyproject.toml``.  The first count loads
the tokenizer once per process:

* ``DEEPSEEK_TOKENIZER_PATH`` — path to a local ``tokenizer.json``.  Set this to
  stay offline and to pin the exact tokenizer file.
* ``DEEPSEEK_TOKENIZER_MODEL`` — Hugging Face repository to download
  ``tokenizer.json`` from; defaults to ``deepseek-ai/DeepSeek-V4.1-Flash``.

When either resource is missing, loading fails, or the request body is a shape
the official encoder cannot represent (for example a ``response_format`` the
reference conversion rejects), counting falls back to the previous
conservative byte-length estimate.  The fallback is never silent: it logs one
warning per reason and every count carries the ``method`` that produced it,
which is recorded in the run record.  An exact count is only ever reported as
exact.

What an exact count covers
--------------------------

The count is the tokenized prompt the reference implementation renders for the
request that is about to be sent: control tokens, message boundaries, the
system prompt, tool declarations, tool calls, tool results, thinking-mode
formatting, and the reasoning-effort prefix.  Two provider-side details cannot
be observed locally and are therefore assumptions, both of which the official
toolkit implements the same way:

* thinking mode is assumed on for requests that do not set it, matching
  ``ConversionOptions.default_thinking_mode`` and DeepSeek's own default;
* tool declarations are omitted from the prompt when a request sets
  ``tool_choice="none"``, as the reference conversion does (one tool schema is
  worth roughly 250 tokens, so a divergence here is small).

A ``response_format`` of ``json_object`` whose prompt text does not contain the
word "json" is rejected by the reference conversion and keeps the conservative
estimate; exploration and fix-generation requests are counted exactly.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from math import ceil
from typing import Any, Mapping

#: Hugging Face repository holding the V4.1 ``tokenizer.json``.
DEFAULT_TOKENIZER_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
#: Environment variable naming a local ``tokenizer.json`` file.
TOKENIZER_PATH_ENV = "DEEPSEEK_TOKENIZER_PATH"
#: Environment variable naming the Hugging Face repository to load from.
TOKENIZER_MODEL_ENV = "DEEPSEEK_TOKENIZER_MODEL"

#: Count method of the official V4.1 prompt encoding: exact for the prompt the
#: reference implementation renders for the request that is about to be sent.
EXACT_METHOD = "deepseek-v4.1-prompt-encoding"
#: Fallback methods.  They are conservative upper bounds, not exact counts.
UNSUPPORTED_REQUEST_METHOD = "byte-heuristic-unsupported-request"
MISSING_RESOURCE_METHOD = "byte-heuristic-resources-unavailable"
OTHER_MODEL_METHOD = "byte-heuristic-non-deepseek-model"

#: Model-name fragments that identify the DeepSeek V4.1 Flash prompt encoding.
_V41_FLASH_MARKERS = (
    "deepseek-v4.1-flash",
    "deepseek-v4-1-flash",
    "deepseek-v41-flash",
    "deepseek-v4.1flash",
    "deepseek-v41flash",
)

#: Characters per token assumed by the conservative fallback estimate.
_FALLBACK_BYTES_PER_TOKEN = 2

_LOGGER = logging.getLogger(__name__)
_NOTIFIED_REASONS: set[str] = set()
_ENCODER: "_RequestEncoder | None" = None
_ENCODER_FAILURE: str | None = None


class TokenCountUnavailable(RuntimeError):
    """The official encoder cannot count the request in this process.

    ``unsupported`` separates a request shape the reference conversion rejects
    from a missing or unusable local resource.
    """

    def __init__(self, message: str, *, unsupported: bool = False) -> None:
        super().__init__(message)
        self.unsupported = unsupported


@dataclass(frozen=True)
class InputTokenCount:
    """One preflight input-token count and the method that produced it."""

    tokens: int
    method: str
    exact: bool


def conservative_byte_estimate(payload: Mapping[str, Any]) -> int:
    """Return the dependency-free upper-bound estimate for a JSON request.

    Two characters per token keeps the estimate above real usage for English
    and source code with the DeepSeek tokenizer, at the cost of refusing some
    requests that would have fit.  It is only reported as ``exact=False``.
    """
    encoded = json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))
    return ceil(len(encoded.encode("utf-8")) / _FALLBACK_BYTES_PER_TOKEN)


def is_deepseek_v41_flash(model: str) -> bool:
    """Return whether ``model`` names the deepseek-v4.1-flash model family."""
    name = (model or "").strip().lower()
    if not name.startswith("deepseek"):
        return False
    return any(marker in name for marker in _V41_FLASH_MARKERS)


class _RequestEncoder:
    """The official V4.1 prompt encoding with a loaded tokenizer attached."""

    def __init__(
        self,
        *,
        request_class: Any,
        options: Any,
        model_version: Any,
        encoding: Any,
    ) -> None:
        self._request_class = request_class
        self._base_options = options
        self._model_version = model_version
        self._encoding = encoding
        self._options_by_model: dict[str, Any] = {}

    def _options_for(self, model: str) -> Any:
        """Return conversion options that accept ``model`` as a V4.1 request."""
        options = self._options_by_model.get(model)
        if options is None:
            if not is_deepseek_v41_flash(model):
                raise TokenCountUnavailable(
                    f"the official V4.1 prompt encoding does not describe model {model!r}"
                )
            options = self._base_options.with_model_alias(model, self._model_version)
            self._options_by_model[model] = options
        return options

    def count(self, body: Mapping[str, Any]) -> int:
        """Return the prompt token count of one complete request body."""
        model = body.get("model")
        if not isinstance(model, str) or not model:
            raise TokenCountUnavailable("the request has no model name to resolve")
        options = self._options_for(model)
        request = self._request_class(dict(body))
        conversation = request.convert(options).conversation
        return len(self._encoding.encode(conversation))


def _load_tokenizer() -> Any:
    """Load the V4.1 tokenizer from a local file or the Hugging Face Hub."""
    from deepseek_recipe import Tokenizer

    path = os.getenv(TOKENIZER_PATH_ENV, "").strip()
    if path:
        return Tokenizer.from_file(path)
    identifier = os.getenv(TOKENIZER_MODEL_ENV, "").strip() or DEFAULT_TOKENIZER_MODEL
    return Tokenizer.from_pretrained(identifier)


def _load_encoder() -> _RequestEncoder:
    """Build the official V4.1 request encoder; raises when unavailable."""
    try:
        from deepseek_recipe import (
            ChatCompletionRequest,
            ConversionOptions,
            DeepseekV41Encoding,
            ModelVersion,
        )
    except ImportError as error:
        raise TokenCountUnavailable(
            f"the deepseek-recipe package is not installed ({error})"
        ) from error
    try:
        tokenizer = _load_tokenizer()
    except Exception as error:  # noqa: BLE001 - any loader failure shares one path.
        raise TokenCountUnavailable(
            f"the DeepSeek V4.1 tokenizer could not be loaded "
            f"({type(error).__name__}: {error})"
        ) from error
    try:
        encoding = DeepseekV41Encoding().with_tokenizer(tokenizer)
    except Exception as error:  # noqa: BLE001 - the bindings reject a bad tokenizer.
        raise TokenCountUnavailable(
            f"the DeepSeek V4.1 prompt encoding rejected the tokenizer "
            f"({type(error).__name__}: {error})"
        ) from error
    return _RequestEncoder(
        request_class=ChatCompletionRequest,
        options=ConversionOptions(),
        model_version=ModelVersion.DeepseekV41Flash,
        encoding=encoding,
    )


def _request_encoder() -> _RequestEncoder:
    """Return the process-wide encoder, loading it at most once."""
    global _ENCODER, _ENCODER_FAILURE
    if _ENCODER is not None:
        return _ENCODER
    if _ENCODER_FAILURE is not None:
        raise TokenCountUnavailable(_ENCODER_FAILURE)
    try:
        encoder = _load_encoder()
    except TokenCountUnavailable as error:
        # Remember the failure so a run does not retry a download per request.
        _ENCODER_FAILURE = str(error)
        raise
    _ENCODER = encoder
    return encoder


def reset_cached_encoder() -> None:
    """Drop the cached encoder so the next count reloads it.

    Used by tests and after installing or downloading tokenizer resources.
    """
    global _ENCODER, _ENCODER_FAILURE
    _ENCODER = None
    _ENCODER_FAILURE = None
    _NOTIFIED_REASONS.clear()


def count_deepseek_request_tokens(payload: Mapping[str, Any]) -> InputTokenCount:
    """Count one complete DeepSeek V4.1 request with the official encoding.

    Raises :class:`TokenCountUnavailable` when the encoder, the tokenizer, or
    the request format is unsupported; it never guesses.
    """
    encoder = _request_encoder()
    try:
        tokens = encoder.count(payload)
    except TokenCountUnavailable as error:
        if error.unsupported:
            raise
        raise TokenCountUnavailable(
            f"the official V4.1 prompt encoding rejected the request ({error})",
            unsupported=True,
        ) from error
    except Exception as error:  # noqa: BLE001 - bindings raise several types.
        raise TokenCountUnavailable(
            f"the official V4.1 prompt encoding rejected the request "
            f"({type(error).__name__}: {error})",
            unsupported=True,
        ) from error
    return InputTokenCount(tokens=tokens, method=EXACT_METHOD, exact=True)


def _notify_once(reason: str, message: str, *, level: int = logging.WARNING) -> None:
    """Log one count notice per reason for the life of the process."""
    if reason in _NOTIFIED_REASONS:
        return
    _NOTIFIED_REASONS.add(reason)
    _LOGGER.log(level, message)


def count_input_tokens(
    payload: Mapping[str, Any], *, provider: str, model: str
) -> InputTokenCount:
    """Count a request before it is sent, exactly when that is possible.

    DeepSeek V4.1 requests served by the Token Harbor provider are counted with
    the official prompt encoding.  Every other request keeps the previous
    conservative byte-length estimate, and every fallback is announced once.
    """
    if provider == "tokenharbor" and is_deepseek_v41_flash(model):
        try:
            return count_deepseek_request_tokens(payload)
        except TokenCountUnavailable as error:
            method = (
                UNSUPPORTED_REQUEST_METHOD
                if error.unsupported
                else MISSING_RESOURCE_METHOD
            )
            count = conservative_byte_estimate(payload)
            _notify_once(
                f"{method}:{error}",
                f"Debuggernaut could not count this DeepSeek V4.1 request with the "
                f"official prompt encoding ({error}); using the conservative "
                f"byte-length estimate of {count} tokens instead. The reported "
                "preflight count is an approximation, not an exact count.",
            )
            return InputTokenCount(tokens=count, method=method, exact=False)
    else:
        method = OTHER_MODEL_METHOD
        count = conservative_byte_estimate(payload)
        _notify_once(
            method,
            f"Debuggernaut has no exact preflight counter for provider "
            f"{provider!r} and model {model!r}; using the conservative "
            f"byte-length estimate of {count} tokens. The reported preflight "
            "count is an approximation, not an exact count.",
            level=logging.INFO,
        )
        return InputTokenCount(tokens=count, method=method, exact=False)
