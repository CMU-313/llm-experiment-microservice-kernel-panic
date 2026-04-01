import os
from typing import Any

import httpx

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:0.6b"


def _ollama_base_url() -> str:
    return os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")


def _ollama_model() -> str:
    return os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)


def _user_prompt(post: str) -> str:
    return f"""Analyze the following text and do two things:
1. Detect if it is English or not.
2. If it is not English, translate it to English. If it is English, repeat it as-is.

Reply in exactly this format and nothing else:
LANGUAGE: <English or name of language>
TRANSLATION: <the English text>

Text: {post}"""


def _parse_model_content(raw: str, post: str) -> tuple[bool, str]:
    content = raw.strip()
    if "</redacted_thinking>" in content:
        content = content.split("</redacted_thinking>")[-1].strip()
    language = None
    translation = post
    for line in content.splitlines():
        if line.startswith("LANGUAGE:"):
            language = line[len("LANGUAGE:") :].strip()
        elif line.startswith("TRANSLATION:"):
            translation = line[len("TRANSLATION:") :].strip()
    if language is None:
        return (True, post)
    return (language.lower() == "english", translation)


def _httpx_timeout() -> httpx.Timeout:
    connect = float(os.environ.get("OLLAMA_TIMEOUT_CONNECT", "5.0"))
    read = float(os.environ.get("OLLAMA_TIMEOUT_READ", "90.0"))
    return httpx.Timeout(connect=connect, read=read, write=10.0, pool=5.0)


def translate_content(content: str) -> tuple[bool, str]:
    return query_llm_robust(content)


def query_llm_robust(post: str) -> tuple[bool, str]:
    url = f"{_ollama_base_url()}/api/chat"
    payload: dict[str, Any] = {
        "model": _ollama_model(),
        "messages": [{"role": "user", "content": _user_prompt(post)}],
        "stream": False,
    }
    try:
        with httpx.Client(timeout=_httpx_timeout()) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception:
        return (True, post)

    message = data.get("message")
    if not isinstance(message, dict):
        return (True, post)
    content = message.get("content")
    if not isinstance(content, str):
        return (True, post)

    return _parse_model_content(content, post)
