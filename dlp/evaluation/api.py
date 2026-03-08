"""
Shared LLM call: OpenAI-compatible API (OpenAI, DeepSeek, etc.) with retries.
"""

import os
import time


def call_openai(messages: list[dict], config: dict) -> str:
    """
    Call an OpenAI-compatible chat API. Retries with backoff on 429 (rate_limit) and 503.
    Does not retry on insufficient_quota (429); raises with a clear message.

    Config:
      - model, temperature, max_tokens: as usual
      - base_url (optional): e.g. "https://api.deepseek.com" for DeepSeek
      - api_key (optional): override; default os.environ["OPENAI_API_KEY"]
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required. pip install openai")

    api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
    base_url = config.get("base_url") or os.environ.get("OPENAI_BASE_URL")
    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)
    model = config.get("model", "gpt-4o-mini")
    temperature = config.get("temperature", 0.0)
    max_tokens = config.get("max_tokens", 1024)

    max_retries = 3
    base_delay = 2.0

    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            err_code = getattr(e, "code", None) or ""
            err_body = str(getattr(e, "body", "") or getattr(e, "message", str(e)))
            is_quota = "insufficient_quota" in err_body or err_code == "insufficient_quota"
            if is_quota:
                raise RuntimeError(
                    "API: insufficient quota / billing limit exceeded (429). "
                    "Use a different key, base_url (e.g. DeepSeek), or run locally with vLLM (see README)."
                ) from e
            is_retryable = (
                "429" in err_body or "rate_limit" in err_body or "503" in err_body
            ) and attempt < max_retries - 1
            if is_retryable:
                delay = base_delay * (2**attempt)
                time.sleep(delay)
                continue
            raise

    raise RuntimeError("API call failed after retries")
