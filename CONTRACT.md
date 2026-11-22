# Contract between core gateway and middlewares

Frozen interface. Any change must be written here in a **separate commit** before it lands.

## Schemas (`gateway/schemas.py`)
- `ChatMessage`, `ChatRequest`, `ChatResponse`, `Usage`, `ChatChoice`: OpenAI chat-completions compatible (extra fields allowed).
- `ChatChunk` / `ChatChunkChoice` / `ChatDelta`: streaming chunk (`chat.completion.chunk`).
- `RequestContext`: `api_key_id`, `tenant_id`, `request_id`, `budget: BudgetInfo`, `extra: dict` (scratchpad for middlewares, e.g. cache keys, PII mapping).
- `BudgetInfo`: `tokens_remaining`, `usd_remaining` (None = unlimited).

## Middleware (`gateway/middleware/base.py`)
```python
class Middleware(Protocol):
    name: str
    async def before(self, req: ChatRequest, ctx: RequestContext) -> ChatRequest | ChatResponse
    async def after(self, req: ChatRequest, resp: ChatResponse, ctx: RequestContext) -> ChatResponse
```
- `before` returning a `ChatResponse` **short-circuits** (e.g. cache hit): the provider and later middlewares are skipped.
- `after` runs in **reverse** order, only for middlewares whose `before` ran (including the one that short-circuited).
- Pipeline: `gateway/pipeline.py::Pipeline(middlewares).run(req, ctx, provider)`.
- Default order: `pii -> cache -> provider`.

## Ownership
- Agent B: `gateway/middleware/pii/`, `gateway/middleware/cache/` (branch `feat/middleware`).
- Agent A: everything else. Each middleware package exposes a class with the Protocol above; the core wires it in.

## Streaming
Middlewares operate on non-streaming `ChatRequest`/`ChatResponse`. For `stream=true` the core runs `before()`, streams the provider, aggregates the full response, then runs `after()` (cache can store it; replaying a cached response as a stream is the core's job).

## Cache metrics
Cache middleware sets `ctx.extra["cache_hit"] = True/False` so the core can export hit rate.
