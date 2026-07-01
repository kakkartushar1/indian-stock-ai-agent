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
    """
    if not api_keys:
        raise LLMConfigurationError("No Mistral API keys configured.")

    last_error: Exception | None = None
    for idx, (key, label) in enumerate(zip(api_keys, key_labels)):
        upper_label = label.upper()
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
            raise

    # All keys exhausted with 429 errors.
    print(
        "[Mistral] All API keys exhausted due to rate limiting (429). "
        "Please wait before retrying or add more API keys.",
        flush=True,
    )
    raise LLMConfigurationError(
        "All Mistral API keys exhausted due to rate limiting (429). "
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


def _build_compatible_model(config: LLMProviderConfig) -> Any:
    if config.disable_tracing:
        set_tracing_disabled(True)

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
