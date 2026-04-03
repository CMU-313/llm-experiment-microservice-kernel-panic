import json

import httpx
import pytest
import respx

from src.translator import (
    _heuristic_non_english,
    _input_is_english,
    _parse_model_content,
    _translation_reads_english,
    _user_prompt,
    query_llm_robust,
    translate_content,
)

_DEFAULT_CHAT_URL = "http://127.0.0.1:11434/api/chat"


@pytest.mark.parametrize(
    ("raw", "post", "expected"),
    [
        (
            "LANGUAGE: English\nTRANSLATION: Hello, world.",
            "Hello, world.",
            (True, "Hello, world.", "English"),
        ),
        (
            "LANGUAGE: French\nTRANSLATION: Good day.",
            "Bonjour.",
            (False, "Good day.", "French"),
        ),
        (
            "TRANSLATION: only this line",
            "some input",
            (True, "some input", None),
        ),
        (
            "</redacted_thinking>\nLANGUAGE: Spanish\nTRANSLATION: Hello.",
            "Hola",
            (False, "Hello.", "Spanish"),
        ),
        (
            "LANGUAGE: German\nTRANSLATION: Hi there",
            "src",
            (False, "Hi there", "German"),
        ),
        (
            "No LANGUAGE line at all.\nJust prose.",
            "orig",
            (True, "orig", None),
        ),
        (
            "LANGUAGE: english\nTRANSLATION: Same",
            "x",
            (True, "Same", "english"),
        ),
        (
            "* LANGUAGE: Italian\nTRANSLATION: Hi",
            "x",
            (False, "Hi", "Italian"),
        ),
    ],
)
def test_parse_model_content(
    raw: str,
    post: str,
    expected: tuple[bool, str, str | None],
) -> None:
    assert _parse_model_content(raw, post) == expected


def test_user_prompt_includes_post_text() -> None:
    text = "unique-marker-xyz"
    assert text in _user_prompt(text)
    assert "LANGUAGE:" in _user_prompt(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", True),
        ("fall", True),
        ("   \t  ", True),
        ("Hello world", True),
        ("This is clearly English text.", True),
        ("Hola mundo, ¿cómo estás?", False),
        ("Me gusta mucho este servicio de traducción", False),
        (
            "El perro del vecino ladra mucho todos los dias",
            False,
        ),
    ],
)
def test_input_is_english_detects_input_language(text: str, expected: bool) -> None:
    assert _input_is_english(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", True),
        ("Hello there friend", True),
        ("Good morning", True),
        ("Hola mundo, ¿cómo estás?", False),
        ("Estoy muy contento hoy", False),
    ],
)
def test_translation_reads_english(text: str, expected: bool) -> None:
    assert _translation_reads_english(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("café", True),
        ("¿Hola?", True),
        ("plain ASCII", False),
        ("niño", True),
    ],
)
def test_heuristic_non_english(text: str, expected: bool) -> None:
    assert _heuristic_non_english(text) is expected


@respx.mock
def test_query_llm_robust_posts_chat_and_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    route = respx.post(_DEFAULT_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "qwen3:0.6b",
                "message": {
                    "role": "assistant",
                    "content": "LANGUAGE: English\nTRANSLATION: out",
                },
                "done": True,
            },
        )
    )

    is_english, text, lang = query_llm_robust("in")
    assert is_english is True
    assert text == "out"
    assert lang == "English"
    assert route.called
    payload = json.loads(route.calls[0].request.content.decode())
    assert payload["model"] == "qwen3:0.6b"
    assert payload["stream"] is False
    assert len(payload["messages"]) == 1
    assert "in" in payload["messages"][0]["content"]


@respx.mock
def test_query_llm_robust_connect_error_returns_original(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    req = httpx.Request("POST", _DEFAULT_CHAT_URL)
    respx.post(_DEFAULT_CHAT_URL).mock(
        side_effect=httpx.ConnectError("refused", request=req),
    )
    assert query_llm_robust("fall") == (True, "fall", None)


@respx.mock
def test_query_llm_robust_missing_message_dict_returns_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    respx.post(_DEFAULT_CHAT_URL).mock(
        return_value=httpx.Response(200, json={"done": True}),
    )
    assert query_llm_robust("z") == (True, "z", None)


@respx.mock
def test_query_llm_robust_non_string_message_content_returns_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    respx.post(_DEFAULT_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": None}, "done": True},
        ),
    )
    assert query_llm_robust("y") == (True, "y", None)


@respx.mock
def test_query_llm_robust_translate_only_when_model_claims_english_but_spanish_translation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model returns LANGUAGE: English with Spanish text; second call must translate."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    spanish = "Hola mundo, ¿cómo estás?"

    def reply(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        user = payload["messages"][0]["content"]
        if "Translate the following into natural English" in user:
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "Hello, how are you?"}},
            )
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": (
                        "LANGUAGE: English\n"
                        f"TRANSLATION: {spanish}"
                    ),
                },
            },
        )

    respx.post(_DEFAULT_CHAT_URL).mock(side_effect=reply)
    is_english, out, lang = query_llm_robust(spanish)
    assert is_english is False
    assert out == "Hello, how are you?"
    assert lang == "English"


@respx.mock
def test_query_llm_robust_translate_only_when_translation_partially_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrong LANGUAGE + wrong TRANSLATION (not equal to post) must still trigger retry."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    post = "Hola mundo, ¿cómo estás?"

    def reply(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        user = payload["messages"][0]["content"]
        if "Translate the following into natural English" in user:
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "Hello, how are you?"}},
            )
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": (
                        "LANGUAGE: English\n"
                        "TRANSLATION: Hola mundo, ¿cómo está?"
                    ),
                },
            },
        )

    respx.post(_DEFAULT_CHAT_URL).mock(side_effect=reply)
    is_english, out, lang = query_llm_robust(post)
    assert is_english is False
    assert out == "Hello, how are you?"
    assert lang == "English"


@respx.mock
def test_query_llm_robust_spanish_label_bad_translation_uses_translate_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    post = "Buenos días"

    def reply(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        user = payload["messages"][0]["content"]
        if "Translate the following into natural English" in user:
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "Good morning"}},
            )
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": (
                        "LANGUAGE: Spanish\n"
                        f"TRANSLATION: {post}"
                    ),
                },
            },
        )

    respx.post(_DEFAULT_CHAT_URL).mock(side_effect=reply)
    is_english, out, lang = query_llm_robust(post)
    assert is_english is False
    assert out == "Good morning"
    assert lang == "Spanish"


def test_translate_content_delegates_to_query_llm_robust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake(post: str) -> tuple[bool, str, str | None]:
        calls.append(post)
        return (True, "ok", None)

    monkeypatch.setattr("src.translator.query_llm_robust", fake)
    assert translate_content("hi") == (True, "ok", None)
    assert calls == ["hi"]


def test_translate_content_strips_html(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    def fake(post: str) -> tuple[bool, str, str | None]:
        captured.append(post)
        return (False, "Hello", "Spanish")

    monkeypatch.setattr("src.translator.query_llm_robust", fake)
    translate_content("<p>Hola</p>")
    assert captured == ["Hola"]
