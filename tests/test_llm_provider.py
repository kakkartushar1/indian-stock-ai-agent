import os
import unittest
from io import StringIO
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
    """Simulated 429 rate-limit error via status_code attribute."""

    status_code = 429


class _MistralBodyRateLimitError(Exception):
    """Simulated Mistral 429 error with the exact body format Mistral returns.

    Error body format:
        {'object': 'error', 'message': 'Rate limit exceeded', 'type': 'rate_limited',
         'param': None, 'code': '1300', 'raw_status_code': 429}
    """

    body = {
        "object": "error",
        "message": "Rate limit exceeded",
        "type": "rate_limited",
        "param": None,
        "code": "1300",
        "raw_status_code": 429,
    }


class _MistralCode1300Error(Exception):
    """Simulated Mistral error with code=1300 but no raw_status_code."""

    body = {"object": "error", "message": "Rate limit exceeded", "code": "1300"}


class _MistralTypeRateLimitedError(Exception):
    """Simulated Mistral error with type='rate_limited' in body."""

    body = {"object": "error", "message": "Rate limit exceeded", "type": "rate_limited"}


class _AuthError(Exception):
    """Simulated non-429 authentication error."""

    status_code = 401


class _ServerError(Exception):
    """Simulated non-429 server error."""

    status_code = 500


class IsRateLimitErrorTests(unittest.TestCase):
    """Unit tests for _is_rate_limit_error covering all detection paths."""

    def test_status_code_429_returns_true(self):
        """Direct status_code=429 attribute triggers rate-limit detection."""
        self.assertTrue(_is_rate_limit_error(_RateLimitError("too many requests")))

    def test_non_429_status_code_returns_false(self):
        """Non-429 status codes do NOT trigger rate-limit detection."""
        self.assertFalse(_is_rate_limit_error(_AuthError("unauthorized")))
        self.assertFalse(_is_rate_limit_error(_ServerError("internal server error")))

    def test_string_fallback_429_in_message(self):
        """'429' in the exception string triggers rate-limit detection."""
        exc = Exception("HTTP 429 rate limit exceeded")
        self.assertTrue(_is_rate_limit_error(exc))

    def test_string_fallback_rate_limit_in_message(self):
        """'rate limit' in the exception string triggers rate-limit detection."""
        exc = Exception("Error: Rate Limit Exceeded")
        self.assertTrue(_is_rate_limit_error(exc))

    def test_mistral_body_raw_status_code_429(self):
        """Mistral error body with raw_status_code=429 triggers rate-limit detection."""
        self.assertTrue(_is_rate_limit_error(_MistralBodyRateLimitError("rate limited")))

    def test_mistral_body_code_1300(self):
        """Mistral error body with code='1300' triggers rate-limit detection."""
        self.assertTrue(_is_rate_limit_error(_MistralCode1300Error("rate limited")))

    def test_mistral_body_type_rate_limited(self):
        """Mistral error body with type='rate_limited' triggers rate-limit detection."""
        self.assertTrue(_is_rate_limit_error(_MistralTypeRateLimitedError("rate limited")))

    def test_response_status_code_429(self):
        """Nested response.status_code=429 triggers rate-limit detection."""
        exc = Exception("HTTP error")
        mock_response = MagicMock()
        mock_response.status_code = 429
        exc.response = mock_response
        self.assertTrue(_is_rate_limit_error(exc))

    def test_top_level_type_rate_limited(self):
        """Top-level 'type' attribute of 'rate_limited' triggers detection."""
        exc = Exception("rate limited")
        exc.type = "rate_limited"  # type: ignore[attr-defined]
        self.assertTrue(_is_rate_limit_error(exc))

    def test_top_level_code_1300(self):
        """Top-level 'code' attribute of '1300' triggers detection."""
        exc = Exception("rate limited")
        exc.code = "1300"  # type: ignore[attr-defined]
        self.assertTrue(_is_rate_limit_error(exc))

    def test_plain_exception_no_rate_limit_returns_false(self):
        """A plain exception with no rate-limit signals returns False."""
        self.assertFalse(_is_rate_limit_error(Exception("something went wrong")))
        self.assertFalse(_is_rate_limit_error(ValueError("invalid value")))


class MistralFallbackTests(unittest.TestCase):
    """Unit tests for _call_mistral_with_fallback."""

    def _make_fn(self, side_effects: list):
        """Return a mock callable whose successive calls raise or return the given values."""
        mock = MagicMock(side_effect=side_effects)
        return mock

    # ------------------------------------------------------------------
    # Key rotation behaviour
    # ------------------------------------------------------------------

    def test_primary_key_success_no_rotation(self):
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

    def test_primary_429_rotates_to_secondary(self):
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

    def test_primary_and_secondary_429_rotates_to_tertiary(self):
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
        fn.assert_any_call("key-primary")
        fn.assert_any_call("key-secondary")
        fn.assert_any_call("key-tertiary")

    def test_all_keys_429_raises_configuration_error(self):
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

    def test_non_429_error_does_not_rotate(self):
        """Primary returns a non-429 error; raises immediately without trying secondary.

        The error is now re-raised as RuntimeError with '[Key: PRIMARY]' context embedded
        in the message so it surfaces in user-facing output.
        """
        fn = self._make_fn([_AuthError("invalid api key")])
        with self.assertRaises(RuntimeError) as ctx:
            _call_mistral_with_fallback(
                fn,
                api_keys=["key-primary", "key-secondary", "key-tertiary"],
                key_labels=["primary", "secondary", "tertiary"],
            )
        # Only the primary key was attempted.
        fn.assert_called_once_with("key-primary")
        # The error message must include the key slot name.
        self.assertIn("[Key: PRIMARY]", str(ctx.exception))

    def test_server_error_does_not_rotate(self):
        """A 500 server error on primary does NOT trigger key rotation.

        The error is now re-raised as RuntimeError with '[Key: PRIMARY]' context embedded
        in the message so it surfaces in user-facing output.
        """
        fn = self._make_fn([_ServerError("internal server error")])
        with self.assertRaises(RuntimeError) as ctx:
            _call_mistral_with_fallback(
                fn,
                api_keys=["key-primary", "key-secondary", "key-tertiary"],
                key_labels=["primary", "secondary", "tertiary"],
            )
        fn.assert_called_once_with("key-primary")
        # The error message must include the key slot name.
        self.assertIn("[Key: PRIMARY]", str(ctx.exception))

    def test_empty_key_list_raises_immediately(self):
        """Empty key list raises LLMConfigurationError immediately."""
        fn = self._make_fn([])
        with self.assertRaises(LLMConfigurationError):
            _call_mistral_with_fallback(fn, api_keys=[], key_labels=[])
        fn.assert_not_called()

    def test_mistral_body_429_triggers_rotation(self):
        """Mistral's exact error body format (raw_status_code: 429) triggers rotation."""
        expected = {"result": "secondary ok"}
        fn = self._make_fn([_MistralBodyRateLimitError("rate limited"), expected])
        result = _call_mistral_with_fallback(
            fn,
            api_keys=["key-primary", "key-secondary"],
            key_labels=["primary", "secondary"],
        )
        self.assertEqual(result, expected)
        self.assertEqual(fn.call_count, 2)

    def test_mistral_code_1300_triggers_rotation(self):
        """Mistral error with code='1300' in body triggers rotation."""
        expected = {"result": "secondary ok"}
        fn = self._make_fn([_MistralCode1300Error("rate limited"), expected])
        result = _call_mistral_with_fallback(
            fn,
            api_keys=["key-primary", "key-secondary"],
            key_labels=["primary", "secondary"],
        )
        self.assertEqual(result, expected)
        self.assertEqual(fn.call_count, 2)

    # ------------------------------------------------------------------
    # Key slot logging verification
    # ------------------------------------------------------------------

    def test_primary_slot_logged_on_attempt(self):
        """[Mistral] Attempting request with PRIMARY key is printed on first attempt."""
        fn = self._make_fn([{"result": "ok"}])
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            _call_mistral_with_fallback(
                fn,
                api_keys=["key-primary"],
                key_labels=["primary"],
            )
            output = mock_stdout.getvalue()
        self.assertIn("[Mistral] Attempting request with PRIMARY key", output)

    def test_secondary_slot_logged_after_primary_429(self):
        """After primary 429, logs show rotation to SECONDARY and attempt with SECONDARY."""
        fn = self._make_fn([_RateLimitError("rate limited"), {"result": "ok"}])
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            _call_mistral_with_fallback(
                fn,
                api_keys=["key-primary", "key-secondary"],
                key_labels=["primary", "secondary"],
            )
            output = mock_stdout.getvalue()
        self.assertIn("[Mistral] Attempting request with PRIMARY key", output)
        self.assertIn("PRIMARY key hit 429 rate limit", output)
        self.assertIn("Rotating to SECONDARY key", output)
        self.assertIn("[Mistral] Attempting request with SECONDARY key", output)

    def test_tertiary_slot_logged_after_secondary_429(self):
        """After primary and secondary 429, logs show rotation to TERTIARY."""
        fn = self._make_fn(
            [_RateLimitError("rate limited"), _RateLimitError("rate limited"), {"result": "ok"}]
        )
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            _call_mistral_with_fallback(
                fn,
                api_keys=["key-primary", "key-secondary", "key-tertiary"],
                key_labels=["primary", "secondary", "tertiary"],
            )
            output = mock_stdout.getvalue()
        self.assertIn("[Mistral] Attempting request with TERTIARY key", output)
        self.assertIn("Rotating to TERTIARY key", output)

    def test_all_keys_exhausted_logged(self):
        """When all keys are exhausted, the exhaustion message is printed and the
        raised LLMConfigurationError includes the key chain in its message."""
        fn = self._make_fn(
            [
                _RateLimitError("rate limited"),
                _RateLimitError("rate limited"),
                _RateLimitError("rate limited"),
            ]
        )
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            with self.assertRaises(LLMConfigurationError) as ctx:
                _call_mistral_with_fallback(
                    fn,
                    api_keys=["key-primary", "key-secondary", "key-tertiary"],
                    key_labels=["primary", "secondary", "tertiary"],
                )
            output = mock_stdout.getvalue()
        self.assertIn("All API keys hit 429 rate limit", output)
        # The raised exception must include the key chain for user-facing output.
        self.assertIn("[All keys exhausted: PRIMARY", str(ctx.exception))

    def test_no_api_key_values_in_logs(self):
        """Actual API key values must NEVER appear in the printed output."""
        fn = self._make_fn([_RateLimitError("rate limited"), {"result": "ok"}])
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            _call_mistral_with_fallback(
                fn,
                api_keys=["secret-key-abc123", "secret-key-xyz789"],
                key_labels=["primary", "secondary"],
            )
            output = mock_stdout.getvalue()
        self.assertNotIn("secret-key-abc123", output)
        self.assertNotIn("secret-key-xyz789", output)

    def test_success_logged_after_attempt(self):
        """Success message is printed after a successful key attempt."""
        fn = self._make_fn([{"result": "ok"}])
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            _call_mistral_with_fallback(
                fn,
                api_keys=["key-primary"],
                key_labels=["primary"],
            )
            output = mock_stdout.getvalue()
        self.assertIn("PRIMARY key succeeded", output)

    # ------------------------------------------------------------------
    # Backward-compatibility: existing test method names preserved
    # ------------------------------------------------------------------

    def test_mistral_fallback_uses_primary_on_success(self):
        """Alias: Primary key works; secondary and tertiary are never called."""
        self.test_primary_key_success_no_rotation()

    def test_mistral_fallback_uses_secondary_on_primary_429(self):
        """Alias: Primary returns 429; secondary succeeds; tertiary never called."""
        self.test_primary_429_rotates_to_secondary()

    def test_mistral_fallback_uses_tertiary_on_secondary_429(self):
        """Alias: Primary and secondary return 429; tertiary succeeds."""
        self.test_primary_and_secondary_429_rotates_to_tertiary()

    def test_mistral_fallback_raises_on_all_429(self):
        """Alias: All three keys return 429; raises LLMConfigurationError."""
        self.test_all_keys_429_raises_configuration_error()

    def test_mistral_fallback_no_rotation_on_non_429(self):
        """Alias: Primary returns a non-429 error; raises immediately without trying secondary."""
        self.test_non_429_error_does_not_rotate()

    def test_mistral_fallback_raises_when_no_keys(self):
        """Alias: Empty key list raises LLMConfigurationError immediately."""
        self.test_empty_key_list_raises_immediately()


if __name__ == "__main__":
    unittest.main()
