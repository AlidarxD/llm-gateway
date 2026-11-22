from gateway.pipeline import Pipeline
from gateway.schemas import ChatChoice, ChatMessage, ChatRequest, ChatResponse, RequestContext

CTX = RequestContext(api_key_id="k", tenant_id="t", request_id="r")


def _resp(text: str) -> ChatResponse:
    return ChatResponse(
        id="x",
        created=0,
        model="m",
        choices=[ChatChoice(message=ChatMessage(role="assistant", content=text))],
    )


class Tag:
    def __init__(self, name: str, log: list[str]) -> None:
        self.name, self.log = name, log

    async def before(self, req, ctx):
        self.log.append(f"before:{self.name}")
        return req

    async def after(self, req, resp, ctx):
        self.log.append(f"after:{self.name}")
        return resp


class Hit(Tag):
    async def before(self, req, ctx):
        await super().before(req, ctx)
        return _resp("cached")


REQ = ChatRequest(model="fake", messages=[ChatMessage(role="user", content="hi")])


async def test_order_and_reverse_after():
    log: list[str] = []

    async def provider(req, ctx):
        log.append("provider")
        return _resp("ok")

    await Pipeline([Tag("a", log), Tag("b", log)]).run(REQ, CTX, provider)
    assert log == ["before:a", "before:b", "provider", "after:b", "after:a"]


async def test_short_circuit_skips_provider_and_later_middlewares():
    log: list[str] = []

    async def provider(req, ctx):
        raise AssertionError("must not be called")

    out = await Pipeline([Tag("a", log), Hit("c", log), Tag("z", log)]).run(REQ, CTX, provider)
    assert out.choices[0].message.content == "cached"
    assert log == ["before:a", "before:c", "after:c", "after:a"]
