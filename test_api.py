"""
Quick BYOLLM provider health check.

Usage:
    python test_api.py
    python test_api.py --model gpt-4o-mini
"""

import argparse
import sys
from dataclasses import replace

import httpx
from dotenv import load_dotenv
from openai import OpenAI

from llm_provider import LLMConfigurationError, get_provider_status, load_llm_config


def mask_key(key: str) -> str:
    if len(key) < 12:
        return "***"
    return f"{key[:8]}...{key[-4:]}"


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Override MODEL_NAME for this health check.")
    args = parser.parse_args()

    try:
        config = load_llm_config()
    except LLMConfigurationError as exc:
        print(f"[FAIL] {exc}")
        return 1

    if args.model:
        config = replace(config, model_name=args.model)

    status = get_provider_status()
    print(f"[INFO] Provider: {config.provider}")
    print(f"[INFO] Base URL host: {status.get('base_url_host') or 'default OpenAI'}")
    print(f"[INFO] Key detected: {mask_key(config.api_key)}")
    print(f"[INFO] Testing model: {config.model_name}")

    try:
        kwargs = dict(
            api_key=config.api_key,
            timeout=20.0,
            max_retries=0,
            default_headers=config.default_headers or None,
        )
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if not config.verify_ssl:
            kwargs["http_client"] = httpx.Client(verify=False)
        client = OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=config.model_name,
            messages=[{"role": "user", "content": "Reply with exactly: API_OK"}],
            max_tokens=16,
        )
        text = (response.choices[0].message.content or "").strip()
        print(f"[OK] API request succeeded. Response: {text!r}")
        return 0
    except Exception as exc:
        print(f"[FAIL] API request failed: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
