import json

import pytest

from gateway.middleware.pii import PIIConfig, PIIMiddleware, PIIMode, StreamRestorer, get_vault
from gateway.pipeline import Pipeline
from gateway.schemas import ChatChoice, ChatMessage, ChatRequest, ChatResponse, RequestContext

IIN = "900101300019"  # placeholder, validity checked below
PHONE = "+7 (701) 123-45-67"


def _valid_iin() -> str:
    import random

    from pii_dataset import make_iin

    return make_iin(random.Random(7))


def ctx(key: str = "k1") -> RequestContext:
    return RequestContext(api_key_id=key, tenant_id="t1", request_id="r1")


def req(text: str) -> ChatRequest:
    return ChatRequest(model="m", messages=[ChatMessage(role="user", content=text)])


def resp(text: str) -> ChatResponse:
    return ChatResponse(
        id="x",
        created=1,
        model="m",
        choices=[ChatChoice(message=ChatMessage(role="assistant", content=text))],
    )


async def test_redact_and_restore_roundtrip() -> None:
    iin = _valid_iin()
    seen: list[str] = []

    async def provider(r: ChatRequest, c: RequestContext) -> ChatResponse:
        seen.append(str(r.messages[0].content))
        return resp(f"Got it: {r.messages[0].content}")

    c = ctx()
    out = await Pipeline([PIIMiddleware()]).run(
        req(f"IIN {iin} phone {PHONE} again {iin}"), c, provider
    )
    assert iin not in seen[0] and PHONE not in seen[0]
    assert seen[0] == "IIN <IIN_1> phone <PHONE_1> again <IIN_1>"  # stable placeholders
    assert out.choices[0].message.content == f"Got it: IIN {iin} phone {PHONE} again {iin}"


async def test_vault_never_leaks_in_repr() -> None:
    iin = _valid_iin()
    c = ctx()
    await PIIMiddleware().before(req(f"iin {iin}"), c)
    assert (
        iin not in repr(c)
        and iin not in str(c.extra)
        and iin not in json.dumps(c.extra["pii_detected"])
    )
    assert get_vault(c) is not None


async def test_block_mode() -> None:
    mw = PIIMiddleware(PIIConfig(per_key={"strict": PIIMode.BLOCK}))
    out = await mw.before(req(f"iin {_valid_iin()}"), ctx("strict"))
    assert isinstance(out, ChatResponse) and out.choices[0].finish_reason == "content_filter"
    clean = await mw.before(req("hello"), ctx("strict"))
    assert isinstance(clean, ChatRequest)


async def test_log_only_passes_through_but_counts() -> None:
    mw = PIIMiddleware(PIIConfig(default_mode=PIIMode.LOG_ONLY))
    iin = _valid_iin()
    c = ctx()
    out = await mw.before(req(f"iin {iin}"), c)
    assert isinstance(out, ChatRequest) and iin in str(out.messages[0].content)
    assert c.extra["pii_detected"] == {"IIN": 1} and get_vault(c) is None


async def test_per_key_modes() -> None:
    mw = PIIMiddleware(PIIConfig(per_key={"a": PIIMode.LOG_ONLY}))
    iin = _valid_iin()
    a = await mw.before(req(iin), ctx("a"))
    b = await mw.before(req(iin), ctx("b"))
    assert isinstance(a, ChatRequest) and iin in str(a.messages[0].content)
    assert isinstance(b, ChatRequest) and iin not in str(b.messages[0].content)


async def test_content_parts_and_tool_calls() -> None:
    iin = _valid_iin()
    msg = ChatMessage(
        role="user",
        content=[
            {"type": "text", "text": f"iin {iin}"},
            {"type": "image_url", "image_url": {"url": "u"}},
        ],
    )
    tool = ChatMessage(
        role="assistant",
        tool_calls=[
            {
                "id": "1",
                "type": "function",
                "function": {"name": "f", "arguments": f'{{"iin": "{iin}"}}'},
            }
        ],
    )
    out = await PIIMiddleware().before(ChatRequest(model="m", messages=[msg, tool]), ctx())
    assert isinstance(out, ChatRequest)
    assert iin not in out.model_dump_json()


def test_stream_restorer_across_chunks() -> None:
    from gateway.middleware.pii import PIIVault

    v = PIIVault()
    ph = v.placeholder("IIN", "123456789012")
    full = f"Your IIN is {ph}, and a < b but also <IIN_9> stays."
    for step in (1, 2, 3, 5, 7):
        r = StreamRestorer(v)
        out = "".join(r.feed(full[i : i + step]) for i in range(0, len(full), step)) + r.flush()
        assert out == "Your IIN is 123456789012, and a < b but also <IIN_9> stays."


def test_stream_restorer_holds_partial_only() -> None:
    from gateway.middleware.pii import PIIVault

    v = PIIVault()
    v.placeholder("PHONE", "+70000000000")
    r = StreamRestorer(v)
    assert r.feed("call <PHO") == "call "
    assert r.feed("NE_1> now") == "+70000000000 now"
    assert r.feed("x <") == "x "
    assert r.flush() == "<"


def test_invalid_numbers_not_redacted() -> None:
    mw = PIIMiddleware()
    text, counts = mw.redact_text("order 123456789012 and card 4111111111111112", None)
    assert not counts and text.startswith("order")


@pytest.mark.parametrize(
    "fmt",
    ["+7 701 123 45 67", "8 (701) 123-45-67", "87011234567", "+77011234567", "8-701-123-45-67"],
)
def test_phone_formats(fmt: str) -> None:
    _, counts = PIIMiddleware().redact_text(f"call {fmt}.", None)
    assert counts == {"PHONE": 1}
