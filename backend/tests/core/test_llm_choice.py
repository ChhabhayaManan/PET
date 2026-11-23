import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from openai import APIConnectionError

from app.core.llm_choice import COOKIE_KWARGS, Choice, decode, encode
from app.core.llm_providers import GEMINI_SKIP_SIGNATURE, PROVIDERS, build_client, classify

KEY = "sk-proj-abcdefghijklmnop1234"
CHOICE = Choice("openai", "gpt-4.1-mini", "https://api.openai.com/v1", KEY)


class StatusError(Exception):
    def __init__(self, status, text=""):
        super().__init__(f"Error code: {status} - {text}")
        self.status_code = status


def test_cookie_round_trip_keeps_the_key_unreadable():
    token = encode(CHOICE)

    assert decode(token) == CHOICE
    assert KEY not in token
    assert decode(token[:-4] + "AAAA") is None
    assert decode("not-a-token") is None
    assert KEY not in repr(CHOICE)


@pytest.mark.parametrize("exc, code", [
    (StatusError(401, "invalid api key"), "invalid_key"),
    (StatusError(403), "invalid_key"),
    (StatusError(402, "payment required"), "out_of_credit"),
    (StatusError(429, "You exceeded your current quota"), "out_of_credit"),
    (StatusError(429, "Rate limit reached"), "rate_limited"),
    (StatusError(429, "Rate limit reached on tokens per minute (TPM): Limit 8000. Please try "
                      "again in 3.1s. Need more tokens? See console.groq.com/settings/billing"),
     "rate_limited"),
    (StatusError(400, "This model's maximum context length is 8192 tokens"), "too_many_tokens"),
    (StatusError(404, "The model `x` does not exist"), "model_not_found"),
    (APIConnectionError(request=httpx.Request("GET", "https://api.openai.com")),
     "provider_unreachable"),
    (StatusError(400, "bad request"), "provider_error"),
])
def test_classify_maps_provider_errors(exc, code):
    error = classify(exc, CHOICE)

    assert error.code == code
    assert error.status_code == 424
    assert error.detail.endswith("Change your key or model in the Model window.")


def test_classify_never_echoes_the_key():
    error = classify(StatusError(401, f"Incorrect API key provided: {KEY}. Also sk-other1234567"), CHOICE)

    assert KEY not in error.provider_message
    assert "sk-other1234567" not in error.provider_message
    assert KEY not in error.detail


def test_classify_keeps_the_providers_own_message():
    exc = StatusError(400, "[{'error': {...}}]")
    exc.body = [{"error": {"code": 400, "status": "INVALID_ARGUMENT",
                           "message": "Function call is missing a thought_signature."}}]

    assert classify(exc, CHOICE).provider_message == "Function call is missing a thought_signature."


def test_gemini_tool_calls_carry_a_thought_signature():
    chat = build_client(Choice("gemini", "gemini-3.8-flash", PROVIDERS["gemini"].base_url, "k"),
                        temperature=0.0, max_tokens=64, timeout=5)
    history = [HumanMessage("hi"),
               AIMessage("", tool_calls=[{"name": "add", "args": {"a": 1}, "id": "c1"}]),
               ToolMessage("2", tool_call_id="c1")]

    call = chat._get_request_payload(history)["messages"][1]["tool_calls"][0]

    assert call["extra_content"]["google"]["thought_signature"] == GEMINI_SKIP_SIGNATURE


def test_cookie_is_http_only_and_kept_for_a_year():
    assert COOKIE_KWARGS["max_age"] == 365 * 86400
    assert COOKIE_KWARGS["httponly"] and COOKIE_KWARGS["samesite"] == "strict"
    assert COOKIE_KWARGS["path"] == "/api" and "secure" not in COOKIE_KWARGS
