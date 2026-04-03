import os
import re
from typing import Any, Optional

import httpx
from langdetect import LangDetectException, detect

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:0.6b"


def _ollama_base_url() -> str:
    return os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")


def _ollama_model() -> str:
    return os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)


def _user_prompt(post: str) -> str:
    return f"""The user message below may be English or another language (e.g. Spanish, French, German).

1. If it is ONLY English (no other language), set LANGUAGE to exactly: English
2. If any part is not English, set LANGUAGE to the primary non-English language name (e.g. Spanish).
3. TRANSLATION must be the full text in clear English. For non-English input, translate every word.
4. Do not claim Spanish or other languages are English.

Reply in exactly this format and nothing else:
LANGUAGE: <language name>
TRANSLATION: <English text only>

Text: {post}"""


def _heuristic_non_english(text: str) -> bool:
    """Small-model hint: Spanish/French/German often use these letters."""
    return bool(re.search(r"[ñáéíóúü¿¡àèìòùâêîôûç]", text, re.IGNORECASE))


def _detect_lang_code(text: str) -> Optional[str]:
    if len(text.strip()) < 3:
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


def _langdetect_reliable(text: str) -> bool:
    """langdetect is noisy on very short strings (e.g. 'fall' -> Welsh)."""
    t = text.strip()
    words = t.split()
    return len(words) >= 2 or len(t) >= 10


def _input_is_english(text: str) -> bool:
    """Whether the user text is English (API: is_english refers to input, not the model)."""
    t = text.strip()
    if not t:
        return True
    if _heuristic_non_english(t):
        return False
    if not _langdetect_reliable(t):
        return True
    code = _detect_lang_code(t)
    if code is None:
        return True
    return code == "en"


def _translation_reads_english(text: str) -> bool:
    """True if the model output is plausibly English (guards wrong LANGUAGE: English)."""
    t = text.strip()
    if not t:
        return True
    if _heuristic_non_english(t):
        return False
    if not _langdetect_reliable(t):
        return True
    code = _detect_lang_code(t)
    if code is None:
        return True
    return code == "en"


def _normalize_response_line(line: str) -> str:
    """Strip list markers, markdown, and leading noise so language:/translation: match."""
    s = line.strip()
    s = re.sub(r"^(\d+\.|[*•-])\s+", "", s)
    while s.startswith("*"):
        s = s[1:].lstrip()
    s = s.lstrip()
    s = s.replace("*", "")
    return s.lstrip()


def _translate_only_prompt(post: str) -> str:
    return (
        "Translate the following into natural English. "
        "Reply with ONLY the English translation, one line, no labels or quotes.\n\n"
        f"{post}"
    )


def _strip_model_preamble(text: str) -> str:
    """Drop text before the structured reply (e.g. Qwen thinking blocks)."""
    text = text.strip()
    for end_tag in ("</redacted_thinking>", "`</redacted_thinking>`", "</redacted_thinking>"):
        if end_tag in text:
            text = text.split(end_tag)[-1].strip()
    return text


def _parse_model_content(raw: str, post: str) -> tuple[bool, str, str | None]:
    content = _strip_model_preamble(raw)
    language: str | None = None
    translation = post
    for line in content.splitlines():
        norm = _normalize_response_line(line)
        low = norm.lower()
        if low.startswith("language:"):
            language = norm.split(":", 1)[1].strip()
            language = language.strip("*`\"' ").strip()
        elif low.startswith("translation:"):
            translation = norm.split(":", 1)[1].strip()
            translation = translation.strip("*`\"' ").strip()
    if language is None:
        return (True, post, None)
    lang_lower = language.lower()
    is_english = lang_lower in ("english", "eng", "en") or lang_lower.startswith("english ")
    if not translation:
        translation = post
    return (is_english, translation, language)


def _httpx_timeout() -> httpx.Timeout:
    connect = float(os.environ.get("OLLAMA_TIMEOUT_CONNECT", "5.0"))
    read = float(os.environ.get("OLLAMA_TIMEOUT_READ", "90.0"))
    return httpx.Timeout(connect=connect, read=read, write=10.0, pool=5.0)


def _strip_html(text: str) -> str:
    """Remove HTML tags so the LLM receives plain text (NodeBB sends HTML)."""
    return re.sub(r"<[^>]+>", "", text).strip()


def _ollama_chat(user_text: str) -> Optional[str]:
    url = f"{_ollama_base_url()}/api/chat"
    payload: dict[str, Any] = {
        "model": _ollama_model(),
        "messages": [{"role": "user", "content": user_text}],
        "stream": False,
    }
    try:
        with httpx.Client(timeout=_httpx_timeout()) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception:
        return None
    message = data.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if not isinstance(content, str):
        return None
    return content


def translate_content(content: str) -> tuple[bool, str, str | None]:
    plain = _strip_html(content) if content else content
    post = (plain or content or "").strip()
    return query_llm_robust(post)


def query_llm_robust(post: str) -> tuple[bool, str, str | None]:
    input_en = _input_is_english(post)
    raw = _ollama_chat(_user_prompt(post))
    if raw is None:
        return (input_en, post, None)

    is_eng, translation, detected_lang = _parse_model_content(raw, post)

    lang_code = _detect_lang_code(post)
    looks_foreign = lang_code is not None and lang_code != "en"
    looks_foreign = looks_foreign or _heuristic_non_english(post)

    def try_translate_only() -> Optional[str]:
        raw2 = _ollama_chat(_translate_only_prompt(post))
        if not raw2:
            return None
        line = _strip_model_preamble(raw2).strip().splitlines()[0].strip()
        line = line.strip("\"'")
        if not line or line == post.strip():
            return None
        return line

    if not is_eng:
        if looks_foreign and not _translation_reads_english(translation):
            alt = try_translate_only()
            if alt:
                return (input_en, alt, detected_lang)
        return (input_en, translation, detected_lang)

    if looks_foreign and (
        translation.strip() == post.strip()
        or not _translation_reads_english(translation)
    ):
        alt = try_translate_only()
        if alt:
            return (input_en, alt, detected_lang)

    return (input_en, translation, detected_lang)
