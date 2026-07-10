"""BYOLLM provider configuration for OpenAI-compatible model backends.

The app's agents are built with the OpenAI Agents SDK. Native OpenAI usage can
keep passing a model name string, while Groq, OpenRouter, Mistral, Ollama, and
custom endpoints use the SDK's chat-completions model adapter.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

import httpx
from dotenv import load_dotenv

from openai_sdk import (
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)

# Import the SDK Model base class for proper inheritance.
# The OpenAI Agents SDK Agent() constructor validates that model is a string,
# a Model instance, or None. _MistralFallbackModel must inherit from Model.
try:
    from openai_sdk import Model as _SdkModel  # type: ignore[attr-defined]
except ImportError:
    _SdkModel = None  # type: ignore[assignment,misc]

load_dotenv()


SUPPORTED_PROVIDERS = {"openai", "groq", "openrouter", "mistral", "ollama", "custom"}


class LLMConfigurationError(RuntimeError):
    """Raised when the selected LLM provider is missing required configuration."""


@dataclass(frozen=True)
class LLMProviderConfig:
    provider: str
    model_name: str
    base_url: str | None
    api_key: str
    default_headers: dict[str, str]
    verify_ssl: bool
    disable_tracing: bool


def _env(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value or default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _required(value: str, env_name: str, provider: str) -> str:
    if not value:
        raise LLMConfigurationError(f"{env_name} is required when LLM_PROVIDER={provider}")
    return value


def _custom_base_url(provider: str, default: str | None = None) -> str | None:
    if provider == "openai":
        return _env("LLM_BASE_URL") or _env("OPENAI_BASE_URL") or default
    return _env("LLM_BASE_URL") or default


def _call_mistral_with_fallback(
    make_request_fn: Callable[[str], Any],
    api_keys: list[str],
    key_labels: list[str],
) -> Any:
    """Try each Mistral API key in order, falling back only on HTTP 429 rate-limit errors.

    Args:
        make_request_fn: A callable that accepts an api_key string and performs the
            Mistral API call, returning the result on success or raising on failure.
        api_keys: Ordered list of API keys to try (primary → secondary → tertiary).
        key_labels: Human-readable labels for each key slot (never the actual key values).

    Returns:
        The result from the first successful call.

    Raises:
        The last 429 error if all keys are exhausted, or any non-429 error immediately.
        Error messages always include the active key slot name (e.g. [Key: PRIMARY])
        so the caller can surface it in user-facing output without exposing key values.
    """
    if not api_keys:
        raise LLMConfigurationError("No Mistral API keys configured.")

    last_error: Exception | None = None
    tried_labels: list[str] = []
    for idx, (key, label) in enumerate(zip(api_keys, key_labels)):
        upper_label = label.upper()
        tried_labels.append(upper_label)
        print(f"[Mistral] Attempting request with {upper_label} key", flush=True)
        logger.info("[Mistral] Attempting request with %s key", upper_label)
        try:
            result = make_request_fn(key)
            print(f"[Mistral] {upper_label} key succeeded", flush=True)
            logger.info("[Mistral] %s key succeeded", upper_label)
            return result
        except Exception as exc:  # noqa: BLE001
            # Only rotate to the next key on rate-limit (429) errors.
            if _is_rate_limit_error(exc):
                # Determine the next key label for the log message.
                next_labels = key_labels[idx + 1 :]
                if next_labels:
                    next_label = next_labels[0].upper()
                    print(
                        f"[Mistral] {upper_label} key hit 429 rate limit. "
                        f"Rotating to {next_label} key...",
                        flush=True,
                    )
                    logger.warning(
                        "[Mistral] %s key hit 429 rate limit. Rotating to %s key.",
                        upper_label,
                        next_label,
                    )
                else:
                    print(
                        f"[Mistral] {upper_label} key hit 429 rate limit. "
                        "No more keys to try.",
                        flush=True,
                    )
                    logger.warning(
                        "[Mistral] %s key hit 429 rate limit. No more keys to try.",
                        upper_label,
                    )
                last_error = exc
                continue
            # All other errors (auth, validation, server errors) propagate immediately.
            print(
                f"[Mistral] {upper_label} key raised a non-429 error ({type(exc).__name__}); "
                "not rotating.",
                flush=True,
            )
            logger.error(
                "[Mistral] %s key raised non-429 error (%s); propagating immediately.",
                upper_label,
                type(exc).__name__,
            )
            # Re-raise with key slot context embedded in the message so callers
            # can surface it in user-facing output (e.g. '[Key: PRIMARY] - ...').
            key_context = f"[Key: {upper_label}]"
            raise RuntimeError(
                f"{key_context} {type(exc).__name__}: {exc}"
            ) from exc

    # All keys exhausted with 429 errors — build a descriptive chain string.
    key_chain = " -> ".join(tried_labels)  # e.g. PRIMARY -> SECONDARY -> TERTIARY
    exhausted_context = f"[All keys exhausted: {key_chain}]"
    print(
        f"[Mistral] {exhausted_context} All API keys hit 429 rate limit. "
        "Please wait before retrying or add more API keys.",
        flush=True,
    )
    logger.warning(
        "[Mistral] %s All API keys hit 429 rate limit.",
        exhausted_context,
    )
    raise LLMConfigurationError(
        f"{exhausted_context} All Mistral API keys exhausted due to rate limiting (429). "
        "Please wait before retrying or add more API keys."
    ) from last_error


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if *exc* represents an HTTP 429 / rate-limit error.

    Handles the exact Mistral error format:
        {'object': 'error', 'message': 'Rate limit exceeded', 'type': 'rate_limited',
         'param': None, 'code': '1300', 'raw_status_code': 429}
    as well as standard OpenAI SDK RateLimitError and httpx status codes.
    """
    # 1. Direct status_code attribute (openai.RateLimitError, httpx.HTTPStatusError, etc.)
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True

    # 2. Nested response.status_code (e.g. httpx.HTTPStatusError)
    response = getattr(exc, "response", None)
    if response is not None:
        resp_status = getattr(response, "status_code", None)
        if resp_status == 429:
            return True

    # 3. Mistral-specific: 'raw_status_code' in the error body dict.
    #    The OpenAI SDK wraps the body in exc.body (a dict) for APIStatusError subclasses.
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        if body.get("raw_status_code") == 429:
            return True
        # Also check 'code': '1300' which Mistral uses for rate-limit errors.
        if str(body.get("code", "")) == "1300":
            return True
        # Mistral error type 'rate_limited'
        err_type_body = str(body.get("type", "") or "").lower()
        if "rate_limit" in err_type_body or err_type_body == "rate_limited":
            return True

    # 4. Top-level 'type' attribute (some SDK versions expose this directly).
    err_type = str(getattr(exc, "type", "") or "").lower()
    if "rate_limit" in err_type or err_type == "rate_limited":
        return True

    # 5. Top-level 'code' attribute: Mistral uses code='1300' for rate limiting.
    err_code = str(getattr(exc, "code", "") or "")
    if err_code == "1300":
        return True

    # 6. Fallback: inspect the string representation for common rate-limit signals.
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "rate_limit" in msg or "rate_limited" in msg


def _call_groq_with_fallback(
    make_request_fn: Callable[[str], Any],
    api_keys: list[str],
    key_labels: list[str],
) -> Any:
    """Try each Groq API key in order, falling back only on HTTP 429 rate-limit errors.

    Args:
        make_request_fn: A callable that accepts an api_key string and performs the
            Groq API call, returning the result on success or raising on failure.
        api_keys: Ordered list of API keys to try (primary → secondary → tertiary).
        key_labels: Human-readable labels for each key slot (never the actual key values).

    Returns:
        The result from the first successful call.

    Raises:
        The last 429 error if all keys are exhausted, or any non-429 error immediately.
        Error messages always include the active key slot name (e.g. [Key: PRIMARY])
        so the caller can surface it in user-facing output without exposing key values.
    """
    if not api_keys:
        raise LLMConfigurationError("No Groq API keys configured.")

    last_error: Exception | None = None
    tried_labels: list[str] = []
    for idx, (key, label) in enumerate(zip(api_keys, key_labels)):
        upper_label = label.upper()
        tried_labels.append(upper_label)
        print(f"[Groq] Attempting request with {upper_label} key", flush=True)
        logger.info("[Groq] Attempting request with %s key", upper_label)
        try:
            result = make_request_fn(key)
            print(f"[Groq] {upper_label} key succeeded", flush=True)
            logger.info("[Groq] %s key succeeded", upper_label)
            return result
        except Exception as exc:  # noqa: BLE001
            # Only rotate to the next key on rate-limit (429) errors.
            if _is_rate_limit_error(exc):
                # Determine the next key label for the log message.
                next_labels = key_labels[idx + 1 :]
                if next_labels:
                    next_label = next_labels[0].upper()
                    print(
                        f"[Groq] {upper_label} key hit 429 rate limit. "
                        f"Rotating to {next_label} key...",
                        flush=True,
                    )
                    logger.warning(
                        "[Groq] %s key hit 429 rate limit. Rotating to %s key.",
                        upper_label,
                        next_label,
                    )
                else:
                    print(
                        f"[Groq] {upper_label} key hit 429 rate limit. "
                        "No more keys to try.",
                        flush=True,
                    )
                    logger.warning(
                        "[Groq] %s key hit 429 rate limit. No more keys to try.",
                        upper_label,
                    )
                last_error = exc
                continue
            # All other errors (auth, validation, server errors) propagate immediately.
            print(
                f"[Groq] {upper_label} key raised a non-429 error ({type(exc).__name__}); "
                "not rotating.",
                flush=True,
            )
            logger.error(
                "[Groq] %s key raised non-429 error (%s); propagating immediately.",
                upper_label,
                type(exc).__name__,
            )
            # Re-raise with key slot context embedded in the message so callers
            # can surface it in user-facing output (e.g. '[Key: PRIMARY] - ...').
            key_context = f"[Key: {upper_label}]"
            raise RuntimeError(
                f"{key_context} {type(exc).__name__}: {exc}"
            ) from exc

    # All keys exhausted with 429 errors — build a descriptive chain string.
    key_chain = " -> ".join(tried_labels)  # e.g. PRIMARY -> SECONDARY -> TERTIARY
    exhausted_context = f"[All keys exhausted: {key_chain}]"
    print(
        f"[Groq] {exhausted_context} All API keys hit 429 rate limit. "
        "Please wait before retrying or add more API keys.",
        flush=True,
    )
    logger.warning(
        "[Groq] %s All API keys hit 429 rate limit.",
        exhausted_context,
    )
    raise LLMConfigurationError(
        f"{exhausted_context} All Groq API keys exhausted due to rate limiting (429). "
        "Please wait before retrying or add more API keys."
    ) from last_error


def load_llm_config() -> LLMProviderConfig:
    provider = _env("LLM_PROVIDER", "openai").lower()
    if provider not in SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise LLMConfigurationError(f"Unsupported LLM_PROVIDER={provider!r}. Supported providers: {supported}")

    model_name = _required(_env("MODEL_NAME", "gpt-4o-mini"), "MODEL_NAME", provider)
    verify_ssl = _env_bool("LLM_VERIFY_SSL", True)
    headers: dict[str, str] = {}

    if provider == "openai":
        base_url = _custom_base_url(provider)
        api_key = _required(_env("OPENAI_API_KEY"), "OPENAI_API_KEY", provider)
    elif provider == "groq":
        base_url = _custom_base_url(provider, "https://api.groq.com/openai/v1")
        # Build the ordered list of fallback keys from the named slots.
        _candidate_keys = [
            _env("GROQ_API_KEY_PRIMARY"),
            _env("GROQ_API_KEY_SECONDARY"),
            _env("GROQ_API_KEY_TERTIARY"),
        ]
        _named_keys = [k for k in _candidate_keys if k]
        if _named_keys:
            # Use the primary key as the effective api_key for config; the fallback
            # helper is used at call time (see _call_groq_with_fallback).
            api_key = _named_keys[0]
        else:
            # Backward-compatible: fall back to the legacy GROQ_API_KEY.
            api_key = _required(_env("GROQ_API_KEY"), "GROQ_API_KEY", provider)
    elif provider == "openrouter":
        base_url = _custom_base_url(provider, "https://openrouter.ai/api/v1")
        api_key = _required(_env("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY", provider)
        site_url = _env("OPENROUTER_SITE_URL")
        app_name = _env("OPENROUTER_APP_NAME", "Indian Stock AI Agent")
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name
    elif provider == "mistral":
        base_url = _custom_base_url(provider, "https://api.mistral.ai/v1")
        # Build the ordered list of fallback keys from the named slots.
        _candidate_keys = [
            _env("MISTRAL_API_KEY_PRIMARY"),
            _env("MISTRAL_API_KEY_SECONDARY"),
            _env("MISTRAL_API_KEY_TERTIARY"),
        ]
        _named_keys = [k for k in _candidate_keys if k]
        if _named_keys:
            # Use the primary key as the effective api_key for config; the fallback
            # helper is used at call time (see _call_mistral_with_fallback).
            api_key = _named_keys[0]
        else:
            # Backward-compatible: fall back to the legacy MISTRAL_API_KEY.
            api_key = _required(_env("MISTRAL_API_KEY"), "MISTRAL_API_KEY", provider)
    elif provider == "ollama":
        base_url = _custom_base_url(provider, _env("OLLAMA_BASE_URL", "http://localhost:11434/v1/"))
        api_key = _env("OLLAMA_API_KEY", "ollama")
    else:
        base_url = _required(_env("LLM_BASE_URL") or _env("OPENAI_BASE_URL"), "LLM_BASE_URL", provider)
        api_key = _required(
            _env("LLM_API_KEY") or _env("OPENAI_API_KEY") or _env("CUSTOM_LLM_API_KEY"),
            "LLM_API_KEY",
            provider,
        )

    disable_tracing = _env_bool("LLM_DISABLE_TRACING", provider != "openai" or bool(base_url))

    return LLMProviderConfig(
        provider=provider,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        default_headers=headers,
        verify_ssl=verify_ssl,
        disable_tracing=disable_tracing,
    )


def _make_openai_client(
    api_key: str,
    base_url: str | None,
    default_headers: dict[str, str] | None,
    verify_ssl: bool,
) -> Any:
    """Create a single AsyncOpenAI client for the given api_key."""
    http_client = httpx.AsyncClient(verify=verify_ssl)
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers or None,
        http_client=http_client,
    )


def _get_mistral_fallback_base() -> type:
    """Return the correct base class for _MistralFallbackModel.

    When the OpenAI Agents SDK is available its ``Agent`` constructor validates
    that the ``model`` argument is a string, a ``Model`` instance, or ``None``.
    To satisfy that check we inherit from ``OpenAIChatCompletionsModel`` which
    itself inherits from the SDK's ``Model`` ABC.

    If the SDK is not available (e.g. during unit tests with stubs) we fall
    back to ``object`` so the class can still be instantiated without error.
    """
    # OpenAIChatCompletionsModel already inherits from the SDK Model ABC,
    # so inheriting from it satisfies isinstance(model, Model).
    if OpenAIChatCompletionsModel is not None:
        return OpenAIChatCompletionsModel
    if _SdkModel is not None:
        return _SdkModel
    return object


class _MistralFallbackModel(_get_mistral_fallback_base()):  # type: ignore[misc]
    """Thin wrapper around OpenAIChatCompletionsModel that rotates Mistral API keys on 429.

    The OpenAI Agents SDK calls ``get_response`` / ``stream_response`` on the model
    object.  This wrapper intercepts those calls, tries each key in order, and embeds
    the active key slot name in any error message so it surfaces in user-facing output.

    Inherits from ``OpenAIChatCompletionsModel`` (which itself inherits from the SDK
    ``Model`` ABC) so that ``isinstance(model, Model)`` is True and the SDK's
    ``Agent`` constructor accepts it without raising a ``TypeError``.
    """

    def __init__(
        self,
        model_name: str,
        api_keys: list[str],
        key_labels: list[str],
        base_url: str | None,
        default_headers: dict[str, str] | None,
        verify_ssl: bool,
    ) -> None:
        if not api_keys:
            raise LLMConfigurationError("No Mistral API keys configured.")
        self._model_name = model_name
        self._api_keys = api_keys
        self._key_labels = [lbl.upper() for lbl in key_labels]
        self._base_url = base_url
        self._default_headers = default_headers
        self._verify_ssl = verify_ssl
        # Pre-build one AsyncOpenAI client per key slot so we can swap
        # self._client on 429 without re-creating clients on every request.
        self._key_clients: list[Any] = [
            _make_openai_client(
                api_key=key,
                base_url=base_url,
                default_headers=default_headers,
                verify_ssl=verify_ssl,
            )
            for key in api_keys
        ]
        # Call super().__init__() with the PRIMARY key's client so that
        # self._client (required by OpenAIChatCompletionsModel) is properly
        # initialised and isinstance(model, Model) is satisfied by the SDK.
        super().__init__(model=model_name, openai_client=self._key_clients[0])

    # ------------------------------------------------------------------
    # Core fallback logic (async)
    # ------------------------------------------------------------------

    async def _call_with_fallback(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Try each key in order; rotate only on 429; embed key slot in error messages.

        On each attempt ``self._client`` is swapped to the appropriate key's
        AsyncOpenAI client so that the parent-class method uses the correct
        credentials without needing a separate per-key model wrapper.
        """
        last_error: Exception | None = None
        tried_labels: list[str] = []

        for idx, (client, label) in enumerate(zip(self._key_clients, self._key_labels)):
            tried_labels.append(label)
            print(f"[Mistral] Attempting request with {label} key", flush=True)
            logger.info("[Mistral] Attempting request with %s key", label)
            try:
                # Swap self._client to the current key's client before delegating
                # to the parent class method so it uses the correct credentials.
                self._client = client
                result = await getattr(super(_MistralFallbackModel, self), method_name)(*args, **kwargs)
                print(f"[Mistral] {label} key succeeded", flush=True)
                logger.info("[Mistral] %s key succeeded", label)
                return result
            except Exception as exc:  # noqa: BLE001
                if _is_rate_limit_error(exc):
                    next_labels = self._key_labels[idx + 1 :]
                    if next_labels:
                        next_label = next_labels[0]
                        print(
                            f"[Mistral] {label} key hit 429 rate limit. "
                            f"Rotating to {next_label} key...",
                            flush=True,
                        )
                        logger.warning(
                            "[Mistral] %s key hit 429 rate limit. Rotating to %s key.",
                            label,
                            next_label,
                        )
                    else:
                        print(
                            f"[Mistral] {label} key hit 429 rate limit. No more keys to try.",
                            flush=True,
                        )
                        logger.warning(
                            "[Mistral] %s key hit 429 rate limit. No more keys to try.", label
                        )
                    last_error = exc
                    continue

                # Non-429 error: propagate immediately with key slot context.
                print(
                    f"[Mistral] {label} key raised a non-429 error ({type(exc).__name__}); "
                    "not rotating.",
                    flush=True,
                )
                logger.error(
                    "[Mistral] %s key raised non-429 error (%s); propagating immediately.",
                    label,
                    type(exc).__name__,
                )
                key_context = f"[Key: {label}]"
                raise RuntimeError(f"{key_context} {type(exc).__name__}: {exc}") from exc

        # All keys exhausted.
        key_chain = " -> ".join(tried_labels)  # e.g. PRIMARY -> SECONDARY -> TERTIARY
        exhausted_context = f"[All keys exhausted: {key_chain}]"
        print(
            f"[Mistral] {exhausted_context} All API keys hit 429 rate limit. "
            "Please wait before retrying or add more API keys.",
            flush=True,
        )
        logger.warning("[Mistral] %s All API keys hit 429 rate limit.", exhausted_context)
        raise LLMConfigurationError(
            f"{exhausted_context} All Mistral API keys exhausted due to rate limiting (429). "
            "Please wait before retrying or add more API keys."
        ) from last_error

    # ------------------------------------------------------------------
    # SDK model interface
    # ------------------------------------------------------------------

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        """Intercept get_response and apply key-rotation fallback."""
        return await self._call_with_fallback("get_response", *args, **kwargs)

    async def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        """Intercept stream_response and apply key-rotation fallback."""
        return await self._call_with_fallback("stream_response", *args, **kwargs)


class _GroqFallbackModel(_get_mistral_fallback_base()):  # type: ignore[misc]
    """Thin wrapper around OpenAIChatCompletionsModel that rotates Groq API keys on 429.

    The OpenAI Agents SDK calls ``get_response`` / ``stream_response`` on the model
    object.  This wrapper intercepts those calls, tries each key in order, and embeds
    the active key slot name in any error message so it surfaces in user-facing output.

    Inherits from ``OpenAIChatCompletionsModel`` (which itself inherits from the SDK
    ``Model`` ABC) so that ``isinstance(model, Model)`` is True and the SDK's
    ``Agent`` constructor accepts it without raising a ``TypeError``.
    """

    def __init__(
        self,
        model_name: str,
        api_keys: list[str],
        key_labels: list[str],
        base_url: str | None,
        default_headers: dict[str, str] | None,
        verify_ssl: bool,
    ) -> None:
        if not api_keys:
            raise LLMConfigurationError("No Groq API keys configured.")
        self._model_name = model_name
        self._api_keys = api_keys
        self._key_labels = [lbl.upper() for lbl in key_labels]
        self._base_url = base_url
        self._default_headers = default_headers
        self._verify_ssl = verify_ssl
        # Pre-build one AsyncOpenAI client per key slot so we can swap
        # self._client on 429 without re-creating clients on every request.
        self._key_clients: list[Any] = [
            _make_openai_client(
                api_key=key,
                base_url=base_url,
                default_headers=default_headers,
                verify_ssl=verify_ssl,
            )
            for key in api_keys
        ]
        # Call super().__init__() with the PRIMARY key's client so that
        # self._client (required by OpenAIChatCompletionsModel) is properly
        # initialised and isinstance(model, Model) is satisfied by the SDK.
        super().__init__(model=model_name, openai_client=self._key_clients[0])

    # ------------------------------------------------------------------
    # Core fallback logic (async)
    # ------------------------------------------------------------------

    async def _call_with_fallback(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Try each key in order; rotate only on 429; embed key slot in error messages.

        On each attempt ``self._client`` is swapped to the appropriate key's
        AsyncOpenAI client so that the parent-class method uses the correct
        credentials without needing a separate per-key model wrapper.
        """
        last_error: Exception | None = None
        tried_labels: list[str] = []

        for idx, (client, label) in enumerate(zip(self._key_clients, self._key_labels)):
            tried_labels.append(label)
            print(f"[Groq] Attempting request with {label} key", flush=True)
            logger.info("[Groq] Attempting request with %s key", label)
            try:
                # Swap self._client to the current key's client before delegating
                # to the parent class method so it uses the correct credentials.
                self._client = client
                result = await getattr(super(_GroqFallbackModel, self), method_name)(*args, **kwargs)
                print(f"[Groq] {label} key succeeded", flush=True)
                logger.info("[Groq] %s key succeeded", label)
                return result
            except Exception as exc:  # noqa: BLE001
                if _is_rate_limit_error(exc):
                    next_labels = self._key_labels[idx + 1 :]
                    if next_labels:
                        next_label = next_labels[0]
                        print(
                            f"[Groq] {label} key hit 429 rate limit. "
                            f"Rotating to {next_label} key...",
                            flush=True,
                        )
                        logger.warning(
                            "[Groq] %s key hit 429 rate limit. Rotating to %s key.",
                            label,
                            next_label,
                        )
                    else:
                        print(
                            f"[Groq] {label} key hit 429 rate limit. No more keys to try.",
                            flush=True,
                        )
                        logger.warning(
                            "[Groq] %s key hit 429 rate limit. No more keys to try.", label
                        )
                    last_error = exc
                    continue

                # Non-429 error: propagate immediately with key slot context.
                print(
                    f"[Groq] {label} key raised a non-429 error ({type(exc).__name__}); "
                    "not rotating.",
                    flush=True,
                )
                logger.error(
                    "[Groq] %s key raised non-429 error (%s); propagating immediately.",
                    label,
                    type(exc).__name__,
                )
                key_context = f"[Key: {label}]"
                raise RuntimeError(f"{key_context} {type(exc).__name__}: {exc}") from exc

        # All keys exhausted.
        key_chain = " -> ".join(tried_labels)  # e.g. PRIMARY -> SECONDARY -> TERTIARY
        exhausted_context = f"[All keys exhausted: {key_chain}]"
        print(
            f"[Groq] {exhausted_context} All API keys hit 429 rate limit. "
            "Please wait before retrying or add more API keys.",
            flush=True,
        )
        logger.warning("[Groq] %s All API keys hit 429 rate limit.", exhausted_context)
        raise LLMConfigurationError(
            f"{exhausted_context} All Groq API keys exhausted due to rate limiting (429). "
            "Please wait before retrying or add more API keys."
        ) from last_error

    # ------------------------------------------------------------------
    # SDK model interface
    # ------------------------------------------------------------------

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        """Intercept get_response and apply key-rotation fallback."""
        return await self._call_with_fallback("get_response", *args, **kwargs)

    async def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        """Intercept stream_response and apply key-rotation fallback."""
        return await self._call_with_fallback("stream_response", *args, **kwargs)


def _build_compatible_model(config: LLMProviderConfig) -> Any:
    if config.disable_tracing:
        set_tracing_disabled(True)

    if config.provider == "mistral":
        # Build the ordered list of fallback keys.
        _candidate_pairs = [
            (_env("MISTRAL_API_KEY_PRIMARY"), "PRIMARY"),
            (_env("MISTRAL_API_KEY_SECONDARY"), "SECONDARY"),
            (_env("MISTRAL_API_KEY_TERTIARY"), "TERTIARY"),
        ]
        _named_pairs = [(k, lbl) for k, lbl in _candidate_pairs if k]
        if len(_named_pairs) > 1:
            # Multiple keys available: use the fallback-aware model wrapper.
            api_keys = [k for k, _ in _named_pairs]
            key_labels = [lbl for _, lbl in _named_pairs]
            return _MistralFallbackModel(
                model_name=config.model_name,
                api_keys=api_keys,
                key_labels=key_labels,
                base_url=config.base_url,
                default_headers=config.default_headers or None,
                verify_ssl=config.verify_ssl,
            )
        # Single key (or legacy MISTRAL_API_KEY): fall through to standard model.

    elif config.provider == "groq":
        # Build the ordered list of fallback keys.
        _candidate_pairs = [
            (_env("GROQ_API_KEY_PRIMARY"), "PRIMARY"),
            (_env("GROQ_API_KEY_SECONDARY"), "SECONDARY"),
            (_env("GROQ_API_KEY_TERTIARY"), "TERTIARY"),
        ]
        _named_pairs = [(k, lbl) for k, lbl in _candidate_pairs if k]
        if len(_named_pairs) > 1:
            # Multiple keys available: use the fallback-aware model wrapper.
            api_keys = [k for k, _ in _named_pairs]
            key_labels = [lbl for _, lbl in _named_pairs]
            return _GroqFallbackModel(
                model_name=config.model_name,
                api_keys=api_keys,
                key_labels=key_labels,
                base_url=config.base_url,
                default_headers=config.default_headers or None,
                verify_ssl=config.verify_ssl,
            )
        # Single key (or legacy GROQ_API_KEY): fall through to standard model.

    http_client = httpx.AsyncClient(verify=config.verify_ssl)
    client = AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        default_headers=config.default_headers or None,
        http_client=http_client,
    )
    return OpenAIChatCompletionsModel(model=config.model_name, openai_client=client)


@lru_cache(maxsize=1)
def get_agent_model() -> Any:
    config = load_llm_config()
    if config.provider == "openai" and not config.base_url:
        return config.model_name
    return _build_compatible_model(config)


def clear_provider_cache() -> None:
    get_agent_model.cache_clear()


def is_llm_configured() -> bool:
    try:
        load_llm_config()
        return True
    except LLMConfigurationError:
        return False


def get_configuration_error() -> str | None:
    try:
        load_llm_config()
        return None
    except LLMConfigurationError as exc:
        return str(exc)


def get_provider_status() -> dict[str, Any]:
    try:
        config = load_llm_config()
    except LLMConfigurationError as exc:
        provider = _env("LLM_PROVIDER", "openai").lower()
        return {
            "configured": False,
            "provider": provider,
            "model": _env("MODEL_NAME", "gpt-4o-mini"),
            "base_url_host": None,
            "api_key_set": False,
            "error": str(exc),
        }

    parsed = urlparse(config.base_url or "")
    return {
        "configured": True,
        "provider": config.provider,
        "model": config.model_name,
        "base_url_host": parsed.netloc or None,
        "api_key_set": bool(config.api_key),
        "tracing_disabled": config.disable_tracing,
        "verify_ssl": config.verify_ssl,
    }
