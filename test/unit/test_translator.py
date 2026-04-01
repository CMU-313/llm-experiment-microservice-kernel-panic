import json

import httpx
import pytest
import respx

from src.translator import (
    _parse_model_content,
    _user_prompt,
    query_llm_robust,
    translate_content,
)

_DEFAULT_CHAT_URL = "http://127.0.0.1:11434/api/chat"


@pytest.mark.parametrize(
    ("raw", "post", "expected_english", "expected_text"),
    [
        (
            "LANGUAGE: English\nTRANSLATION: Hello, world.",
            "Hello, world.",
            True,
            "Hello, world.",
        ),
        (
            "LANGUAGE: French\nTRANSLATION: Good day.",
            "Bonjour.",
            False,
            "Good day.",
        ),
        (
            "TRANSLATION: only this line",
            "some input",
            True,
            "some input",
        ),
        (
            "</redacted_thinking>\nLANGUAGE: Spanish\nTRANSLATION: Hello.",
            "Hola",
            False,
            "Hello.",
        ),
        (
            "LANGUAGE: German\nTRANSLATION: Hi there",
            "src",
            False,
            "Hi there",
        ),
        (
            "No LANGUAGE line at all.\nJust prose.",
            "orig",
            True,
            "orig",
        ),
        (
            "LANGUAGE: english\nTRANSLATION: Same",
            "x",
            True,
            "Same",
        ),
    ],
)
def test_parse_model_content(
    raw: str,
    post: str,
    expected_english: bool,
    expected_text: str,
) -> None:
    assert _parse_model_content(raw, post) == (expected_english, expected_text)


def test_user_prompt_includes_post_text() -> None:
    text = "unique-marker-xyz"
    assert text in _user_prompt(text)
    assert "LANGUAGE:" in _user_prompt(text)


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

    is_english, text = query_llm_robust("in")
    assert is_english is True
    assert text == "out"
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
    assert query_llm_robust("fall") == (True, "fall")


@respx.mock
def test_query_llm_robust_missing_message_dict_returns_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    respx.post(_DEFAULT_CHAT_URL).mock(
        return_value=httpx.Response(200, json={"done": True}),
    )
    assert query_llm_robust("z") == (True, "z")


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
    assert query_llm_robust("y") == (True, "y")


def test_translate_content_delegates_to_query_llm_robust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake(post: str) -> tuple[bool, str]:
        calls.append(post)
        return (True, "ok")

    monkeypatch.setattr("src.translator.query_llm_robust", fake)
    assert translate_content("hi") == (True, "ok")
    assert calls == ["hi"]
