"""Hugging Face Inference Providers client, OpenAI-compatible shape.

The original file called the HF router inline inside chat(). Pulling it
out here means: one place to change the model, one place to add retries,
and db.py's summarizer can call it without importing Flask.
"""

import requests

from config import (
    HF_API_URL,
    HF_MODEL,
    HF_TOKEN,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT_SECONDS,
)


class LLMError(Exception):
    """Raised when the LLM call fails in a way the caller should surface."""


def call_llm(messages, tools=None, system=None, max_tokens=LLM_MAX_TOKENS):
    """Call Hugging Face's OpenAI-compatible chat completions endpoint.

    `messages` should already be OpenAI-shaped: role in
    {system, user, assistant, tool}. Returns the raw response JSON.
    """
    if not HF_TOKEN:
        raise LLMError(
            "HF_TOKEN is not set. Export it before starting MIAH "
            "(see README)."
        )

    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    payload = {
        "model": HF_MODEL,
        "messages": full_messages,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools

    try:
        resp = requests.post(
            HF_API_URL,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
    except requests.HTTPError as e:
        # Surface the body — HF puts the real reason (bad model name, rate
        # limit, tool-calling unsupported) in the response body, not the
        # status line, and the original swallowed it.
        body = ""
        try:
            body = e.response.text[:500]
        except Exception:
            pass
        raise LLMError(f"LLM request failed ({e.response.status_code}): {body}") from e
    except requests.RequestException as e:
        raise LLMError(f"Couldn't reach the LLM: {e}") from e

    return resp.json()


def anthropic_tools_to_openai(tools):
    """Tool definitions are written in Anthropic's {name, description,
    input_schema} shape (easier to read), and converted to OpenAI's
    {type, function: {...}} shape only at call time."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]