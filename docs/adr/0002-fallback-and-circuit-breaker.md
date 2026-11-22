# 2. Retries, per-provider circuit breaker, fallback chains

**Status:** accepted

**Context.** Providers time out, rate-limit and fail. Naive retries amplify outages; no retries leak every blip to users.

**Decision.** Per target: bounded retries with *full-jitter* exponential backoff (`uniform(0, min(cap, base·2^n))`, `Retry-After` honoured and capped). Failures feed a per-provider consecutive-failure breaker (closed → open → half-open with a single probe). When retries are exhausted or the breaker is open, the router moves to the next target of the alias's chain. 4xx other than 429/401/403 are the caller's fault: no retry, no fallback, no breaker penalty. Streaming can only fail over before the first chunk, since after that bytes are already with the client.

**Alternatives.** Rolling error-rate breaker (more robust to low traffic noise, more state); hedged requests (doubles cost).

**Consequences.** Verified by integration tests with a fake provider that injects timeouts/429/500, and by a 20%-failure load test with zero client errors. The breaker is per process, so with N workers each learns separately.
