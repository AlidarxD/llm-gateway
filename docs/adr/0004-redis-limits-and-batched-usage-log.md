# 4. Redis for limits, batched inserts for the usage log

**Status:** accepted

**Context.** Rate limits and budgets are checked on every request and shared by all workers; the usage log is write-heavy and must not slow requests down.

**Decision.** Limits live in Redis: a fixed-window `INCR` per key per minute, and daily token / micro-USD counters (`INCRBY`, 48 h TTL). The per-request usage row goes to PostgreSQL via an in-process queue flushed in batches (every 0.5 s or 200 rows) and drained on shutdown. This replaced one commit per request, which capped a worker near 150 req/s.

**Trade-offs.** Fixed windows allow a 2x burst at a boundary (a sliding window or token bucket in Lua would fix it). Budgets are checked before and charged after, so concurrent requests can overshoot. A crash loses at most ~0.5 s of usage rows; the Redis counters (which enforce budgets) are updated immediately.
