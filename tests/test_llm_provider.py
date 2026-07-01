import os
import unittest
from unittest.mock import MagicMock, call, patch

import llm_provider
from llm_provider import (
    LLMConfigurationError,
    _call_mistral_with_fallback,
    _is_rate_limit_error,
    clear_provider_cache,
    load_llm_config,
)


class LLMProviderConfigTests(unittest.TestCase):
    def setUp(self):
        clear_provider_cache()

    def tearDown(self):
        clear_provider_cache()

    def env(self, **values):
        base = {
            "LLM_PROVIDER": "",
            "MODEL_NAME": "",
            "LLM_BASE_URL": "",
            "LLM_API_KEY": "",
            "OPENAI_API_KEY": "",
            "OPENAI_BASE_URL": "",
            "GROQ_API_KEY": "",
            "OPENROUTER_API_KEY": "",
            "OPENROUTER_SITE_URL": "",
            "OPENROUTER_APP_NAME": "",
            "MISTRAL_API_KEY": "",
            "OLLAMA_BASE_URL": "",
            "OLLAMA_API_KEY": "",
            "LLM_VERIFY_SSL": "",
            "LLM_DISABLE_TRACING": "",
        }
        base.update(values)
        return patch.dict(os.environ, base, clear=False)

    def test_default_openai_uses_model_string(self):
        with self.env(OPENAI_API_KEY="sk-test", MODEL_NAME="gpt-4o-mini"):
            self.assertEqual(llm_provider.get_agent_model(), "gpt-4o-mini")

    def test_groq_requires_provider_key(self):
        with self.env(LLM_PROVIDER="groq", MODEL_NAME="llama-3.3-70b-versatile"):
            with self.assertRaisesRegex(LLMConfigurationError, "GROQ_API_KEY"):
                load_llm_config()

    def test_groq_defaults_to_openai_compatible_base_url(self):
        with self.env(
            LLM_PROVIDER="groq",
            MODEL_NAME="llama-3.3-70b-versatile",
            GROQ_API_KEY="gsk-test",
        ):
            config = load_llm_config()
            self.assertEqual(config.base_url, "https://api.groq.com/openai/v1")
            self.assertTrue(config.disable_tracing)

    def test_openrouter_adds_optional_attribution_headers(self):
        with self.env(
            LLM_PROVIDER="openrouter",
            MODEL_NAME="openai/gpt-4o-mini",
            OPENROUTER_API_KEY="or-test",
            OPENROUTER_SITE_URL="https://example.test",
            OPENROUTER_APP_NAME="Stock App",
        ):
            config = load_llm_config()
            self.assertEqual(config.base_url, "https://openrouter.ai/api/v1")
            self.assertEqual(config.default_headers["HTTP-Referer"], "https://example.test")
            self.assertEqual(config.default_headers["X-Title"], "Stock App")

    def test_mistral_defaults_to_v1_base_url(self):
        with self.env(
            LLM_PROVIDER="mistral",
            MODEL_NAME="mistral-large-latest",
            MISTRAL_API_KEY="mis-test",
        ):
            self.assertEqual(load_llm_config().base_url, "https://api.mistral.ai/v1")

    def test_ollama_uses_local_base_url_and_dummy_key(self):
        with self.env(LLM_PROVIDER="ollama", MODEL_NAME="llama3.1"):
            config = load_llm_config()
            self.assertEqual(config.base_url, "http://localhost:11434/v1/")
            self.assertEqual(config.api_key, "ollama")

    def test_legacy_openai_base_url_uses_compatible_model(self):
        with self.env(
            OPENAI_API_KEY="sk-test",
            OPENAI_BASE_URL="https://proxy.example/v1",
            MODEL_NAME="custom-model",
        ):
            with patch.object(llm_provider, "AsyncOpenAI") as async_client, patch.object(
                llm_provider, "OpenAIChatCompletionsModel"
            ) as model_cls:
                model_cls.return_value = "compatible-model"
                self.assertEqual(llm_provider.get_agent_model(), "compatible-model")
                async_client.assert_called_once()
                model_cls.assert_called_once()

    def test_status_never_exposes_api_key(self):
        with self.env(LLM_PROVIDER="mistral", MODEL_NAME="mistral-small", MISTRAL_API_KEY="secret-value"):
            status = llm_provider.get_provider_status()
            self.assertTrue(status["api_key_set"])
            self.assertNotIn("secret-value", repr(status))


class _RateLimitError(Exception):
    """Simulated 429 rate-limit error."""

    status_code = 429


class _AuthError(Exception):
    """Simulated non-429 authentication error."""

    status_code = 401


class MistralFallbackTests(unittest.TestCase):
    """Unit tests for _call_mistral_with_fallback and _is_rate_limit_error."""

    # ------------------------------------------------------------------
    # _is_rate_limit_error helpers
    # ------------------------------------------------------------------

    def test_is_rate_limit_error_status_code_429(self):
        self.assertTrue(_is_rate_limit_error(_RateLimitError("too many requests")))

    def test_is_rate_limit_error_non_429_returns_false(self):
        self.assertFalse(_is_rate_limit_error(_AuthError("unauthorized")))

    def test_is_rate_limit_error_string_fallback(self):
        exc = Exception("HTTP 429 rate limit exceeded")
        self.assertTrue(_is_rate_limit_error(exc))

    # ------------------------------------------------------------------
    # _call_mistral_with_fallback behaviour
    # ------------------------------------------------------------------

    def _make_fn(self, side_effects: list):
        """Return a mock callable whose successive calls raise or return the given values."""
        mock = MagicMock(side_effect=side_effects)
        return mock

    def test_mistral_fallback_uses_primary_on_success(self):
        """Primary key works; secondary and tertiary are never called."""
        expected = {"result": "ok"}
        fn = self._make_fn([expected])
        result = _call_mistral_with_fallback(
            fn,
            api_keys=["key-primary", "key-secondary", "key-tertiary"],
            key_labels=["primary", "secondary", "tertiary"],
        )
        self.assertEqual(result, expected)
        fn.assert_called_once_with("key-primary")

    def test_mistral_fallback_uses_secondary_on_primary_429(self):
        """Primary returns 429; secondary succeeds; tertiary never called."""
        expected = {"result": "secondary ok"}
        fn = self._make_fn([_RateLimitError("rate limited"), expected])
        result = _call_mistral_with_fallback(
            fn,
            api_keys=["key-primary", "key-secondary", "key-tertiary"],
            key_labels=["primary", "secondary", "tertiary"],
        )
        self.assertEqual(result, expected)
        self.assertEqual(fn.call_count, 2)
        fn.assert_any_call("key-primary")
        fn.assert_any_call("key-secondary")

    def test_mistral_fallback_uses_tertiary_on_secondary_429(self):
        """Primary and secondary return 429; tertiary succeeds."""
        expected = {"result": "tertiary ok"}
        fn = self._make_fn(
            [_RateLimitError("rate limited"), _RateLimitError("rate limited"), expected]
        )
        result = _call_mistral_with_fallback(
            fn,
            api_keys=["key-primary", "key-secondary", "key-tertiary"],
            key_labels=["primary", "secondary", "tertiary"],
        )
        self.assertEqual(result, expected)
        self.assertEqual(fn.call_count, 3)

    def test_mistral_fallback_raises_on_all_429(self):
        """All three keys return 429; raises LLMConfigurationError."""
        fn = self._make_fn(
            [
                _RateLimitError("rate limited"),
                _RateLimitError("rate limited"),
                _RateLimitError("rate limited"),
            ]
        )
        with self.assertRaises(LLMConfigurationError) as ctx:
            _call_mistral_with_fallback(
                fn,
                api_keys=["key-primary", "key-secondary", "key-tertiary"],
                key_labels=["primary", "secondary", "tertiary"],
            )
        self.assertIn("exhausted", str(ctx.exception).lower())
        self.assertEqual(fn.call_count, 3)

    def test_mistral_fallback_no_rotation_on_non_429(self):
        """Primary returns a non-429 error; raises immediately without trying secondary."""
        fn = self._make_fn([_AuthError("invalid api key")])
        with self.assertRaises(_AuthError):
            _call_mistral_with_fallback(
                fn,
                api_keys=["key-primary", "key-secondary", "key-tertiary"],
                key_labels=["primary", "secondary", "tertiary"],
            )
        # Only the primary key was attempted.
        fn.assert_called_once_with("key-primary")

    def test_mistral_fallback_raises_when_no_keys(self):
        """Empty key list raises LLMConfigurationError immediately."""
        fn = self._make_fn([])
        with self.assertRaises(LLMConfigurationError):
            _call_mistral_with_fallback(fn, api_keys=[], key_labels=[])
        fn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
