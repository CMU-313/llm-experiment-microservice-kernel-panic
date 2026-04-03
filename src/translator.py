import os
import re
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


def _normalize_response_line(line: str) -> str:
    """Strip list markers and leading markdown so LANGUAGE:/TRANSLATION: can be found."""
    s = line.strip()
    s = re.sub(r"^(\d+\.|[*•-])\s+", "", s)
    while s.startswith("*"):
        s = s[1:].lstrip()
    return s.lstrip()


def _parse_model_content(raw: str, post: str) -> tuple[bool, str, str | None]:
    content = raw.strip()
    if "</redacted_thinking>" in content:
        content = content.split("</redacted_thinking>")[-1].strip()
    detected_language: str | None = None
    translation = post
    for line in content.splitlines():
        norm = _normalize_response_line(line)
        low = norm.lower()
        if low.startswith("language:"):
            detected_language = norm.split(":", 1)[1].strip()
            detected_language = detected_language.strip("*").strip()
        elif low.startswith("translation:"):
            translation = norm.split(":", 1)[1].strip()
            translation = translation.strip("*").strip()
    if detected_language is None:
        return (True, post, None)
    is_english = detected_language.lower() == "english"
    return (is_english, translation, detected_language)


def _httpx_timeout() -> httpx.Timeout:
    connect = float(os.environ.get("OLLAMA_TIMEOUT_CONNECT", "5.0"))
    read = float(os.environ.get("OLLAMA_TIMEOUT_READ", "90.0"))
    return httpx.Timeout(connect=connect, read=read, write=10.0, pool=5.0)


def _strip_html(text: str) -> str:
    """Remove HTML tags so the LLM receives plain text (NodeBB sends HTML)."""
    return re.sub(r"<[^>]+>", "", text).strip()


def translate_content(content: str) -> tuple[bool, str, str | None]:
    plain = _strip_html(content) if content else content
    return query_llm_robust(plain or content)


def query_llm_robust(post: str) -> tuple[bool, str, str | None]:
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
        return (True, post, None)

    message = data.get("message")
    if not isinstance(message, dict):
        return (True, post, None)
    content = message.get("content")
    if not isinstance(content, str):
        return (True, post, None)

    return _parse_model_content(content, post)
