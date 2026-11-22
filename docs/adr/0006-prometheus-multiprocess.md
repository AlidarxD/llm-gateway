# 6. Prometheus multiprocess mode for multi-worker deployments

**Status:** accepted

**Context.** One worker tops out near 150 req/s, so the container runs several uvicorn workers. Per-process registries make each scrape hit a random worker and counters appear to jump backwards.

**Decision.** `PROMETHEUS_MULTIPROC_DIR` is set in the image; `/metrics` aggregates all workers through `MultiProcessCollector`. The breaker gauge uses `multiprocess_mode="max"`. Verified: repeated scrapes of a 4-worker container return identical totals.

**Consequences.** The directory is wiped at container start. Breaker state itself stays per worker (see ADR 2).
