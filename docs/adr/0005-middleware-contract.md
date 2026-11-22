# 5. Middleware contract with short-circuit

**Status:** accepted

**Context.** PII redaction and caching were built in parallel by a separate agent and needed a stable seam.

**Decision.** `Middleware.before(req, ctx) -> ChatRequest | ChatResponse` (a response short-circuits) and `after(req, resp, ctx) -> ChatResponse`, run in order / reverse order, see [CONTRACT.md](../../CONTRACT.md). Order is `pii → cache → provider` so only redacted text is hashed, embedded or stored, and `after()` restores PII last. Per-request state travels in `RequestContext.extra`.

**Consequences.** Middlewares are unit-testable in isolation and the core stays unaware of them. Streaming is handled by the core (aggregate → `after()`, replay hits as chunks) rather than by a streaming-aware middleware API, which keeps the interface small at the cost of middlewares not seeing individual deltas; PII uses an explicit `stream_restorer` hook for that one need.
