"""Adapter tests against httpx.MockTransport (no network, no paid calls)."""

import json

import httpx
import pytest

from gateway.errors import (
    ProviderAuthError,
    ProviderBadRequest,
    ProviderRateLimited,
    ProviderServerError,
    ProviderTimeout,
)
from gateway.providers.anthropic import AnthropicProvider
from gateway.providers.openai_compat import OpenAICompatProvider
from gateway.schemas import ChatMessage, ChatRequest

REQ = ChatRequest(
    model="smart",
    messages=[
        ChatMessage(role="system", content="be brief"),
        ChatMessage(role="user", content="hi"),
    ],
    max_tokens=50,
)

OAI_OK = {
    "id": "c1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "yo"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}


def oai(handler):
    c = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAICompatProvider("openai", "http://x/v1", c, "k")


async def test_openai_complete_rewrites_model_and_auth():
    seen = {}

    def h(request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json=OAI_OK)

    r = await oai(h).complete(REQ, "gpt-4o-mini")
    assert seen["body"]["model"] == "gpt-4o-mini" and seen["auth"] == "Bearer k"
    assert r.usage.total_tokens == 4


@pytest.mark.parametrize(
    "status,exc",
    [
        (429, ProviderRateLimited),
        (500, ProviderServerError),
        (503, ProviderServerError),
        (401, ProviderAuthError),
        (400, ProviderBadRequest),
    ],
)
async def test_openai_status_mapping(status, exc):
    with pytest.raises(exc):
        await oai(lambda r: httpx.Response(status, text="x")).complete(REQ, "m")


async def test_openai_retry_after_parsed():
    p = oai(lambda r: httpx.Response(429, headers={"retry-after": "3"}, text="slow down"))
    with pytest.raises(ProviderRateLimited) as e:
        await p.complete(REQ, "m")
    assert e.value.retry_after == 3.0


async def test_openai_timeout_and_transport_errors():
    def boom(request):
        raise httpx.ReadTimeout("t")

    with pytest.raises(ProviderTimeout):
        await oai(boom).complete(REQ, "m")

    def conn(request):
        raise httpx.ConnectError("c")

    with pytest.raises(ProviderServerError):
        await oai(conn).complete(REQ, "m")


async def test_openai_stream_parsing():
    def chunk(txt, usage=None):
        d = {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": {"content": txt}}],
        }
        if usage:
            d["usage"] = usage
        return "data: " + json.dumps(d) + "\n\n"

    body = (
        chunk("a")
        + chunk("b", {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})
        + "data: [DONE]\n\n"
    )
    p = oai(lambda r: httpx.Response(200, text=body))
    out = [c async for c in p.stream(REQ, "m")]
    assert [c.choices[0].delta.content for c in out] == ["a", "b"] and out[
        -1
    ].usage.total_tokens == 3


def anth(handler):
    c = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return AnthropicProvider(c, "k", "http://a")


async def test_anthropic_translation():
    seen = {}

    def h(request):
        seen["body"] = json.loads(request.content)
        seen["headers"] = request.headers
        return httpx.Response(
            200,
            json={
                "id": "m1",
                "model": "claude-x",
                "stop_reason": "max_tokens",
                "content": [{"type": "text", "text": "hello"}],
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
        )

    r = await anth(h).complete(REQ, "claude-x")
    b = seen["body"]
    assert b["system"] == "be brief" and b["messages"] == [{"role": "user", "content": "hi"}]
    assert b["max_tokens"] == 50 and seen["headers"]["x-api-key"] == "k"
    assert r.choices[0].message.content == "hello" and r.choices[0].finish_reason == "length"
    assert r.usage.total_tokens == 7


async def test_anthropic_stream():
    ev = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 4}}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "he"}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "llo"}},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 3},
        },
    ]
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in ev)
    out = [c async for c in anth(lambda r: httpx.Response(200, text=body)).stream(REQ, "m")]
    assert "".join(c.choices[0].delta.content or "" for c in out) == "hello"
    assert out[-1].usage.total_tokens == 7 and out[-1].choices[0].finish_reason == "stop"


async def test_anthropic_error_mapping():
    with pytest.raises(ProviderRateLimited):
        await anth(lambda r: httpx.Response(429, text="x")).complete(REQ, "m")
