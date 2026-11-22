# llm-gateway

An OpenAI-compatible API gateway that sits between your apps and LLM providers. Point any OpenAI SDK at it and get **provider fallback, per-key rate limits and budgets, PII redaction, response caching, cost tracking and observability** without touching application code.

```
app ──► llm-gateway ──► anthropic / openai / openrouter / ollama / fake
        auth · limits · PII · cache · retries · breaker · fallback · metrics
```

## Features

| Area | What you get |
|---|---|
| API | `POST /v1/chat/completions` (JSON and SSE streaming), `GET /v1/models`, `/healthz`, `/readyz`, `/metrics` |
| Providers | Anthropic (native Messages API, translated), OpenAI, OpenRouter, Ollama, and a `fake` provider with failure injection for tests/demos |
| Reliability | per-call timeouts, retries with full-jitter backoff (honours `Retry-After`), per-provider circuit breaker, automatic fallback chain per model alias |
| Access control | hashed API keys, per-key requests/minute, daily token and USD budgets (Redis) |
| Accounting | per-request usage row in PostgreSQL (tokens, cost, latency, provider, fallbacks, cache hit) |
| Middlewares | PII redaction (KZ/RU checksum-validated IIN/BIN/card/IBAN/phone/...) and exact + opt-in semantic cache, see [CONTRACT.md](CONTRACT.md) |
| Observability | JSON logs with request ids, Prometheus metrics, Grafana dashboard, docker-compose stack |

Model aliases map to fallback chains, e.g. `smart → [anthropic, openrouter]`, `fast → [ollama, openai]` ([config/routes.full.json](config/routes.full.json)). `provider/model` (e.g. `ollama/llama3.2`) bypasses aliases.

## Quickstart (no paid accounts)

```bash
docker compose up -d --build            # app, postgres, redis, prometheus, grafana (fake + Ollama providers)
docker compose exec app python -m gateway.cli create-key --tenant acme --name dev --rpm 120
# prints:  API key (shown once): sk-gw-...

curl localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-gw-..." -H 'content-type: application/json' \
  -d '{"model":"smart","messages":[{"role":"user","content":"hello"}]}'
```

With the stock OpenAI SDK: `OpenAI(base_url="http://localhost:8000/v1", api_key="sk-gw-...")`.

Grafana: <http://localhost:3000> (admin / `$GRAFANA_ADMIN_PASSWORD`, default `admin`), dashboard "LLM Gateway". Prometheus: <http://localhost:9090>.

Configuration is via `GATEWAY_*` environment variables, see [.env.example](.env.example) (export them in your shell; compose substitutes them). Real provider keys are only needed for the "full" routing profile.

Local development:

```bash
uv sync
uv run pytest -q && uv run ruff check . && uv run mypy gateway
```

### Failure drill

```bash
GATEWAY_ENABLED_PROVIDERS='["fake","fake2"]' GATEWAY_ROUTES_FILE=config/routes.loadtest.json \
GATEWAY_FAKE_FAILURE_RATE=0.2 docker compose up -d --build app
```

20% of calls to the primary `fake` provider now fail with random timeouts/429/500; clients still see 200s, and `gateway_retries_total`, `gateway_fallbacks_total` and the breaker gauge show the work being done.

## Load test

`loadtest/locustfile.py`, 100 concurrent users for 30 s, 90% non-streaming / 10% SSE, 4 uvicorn workers, full pipeline (auth → rate limit → PII → cache → router → usage log), upstream = fake provider with 20 ms latency and **20% injected failures** (timeouts/429/500) with a healthy fallback. Locust runs on the same 12-core laptop as the stack, so treat these as a lower bound.

| Scenario | Requests | Failures | Throughput | p50 | p95 | p99 |
|---|---|---|---|---|---|---|
| Cache miss (unique prompts, with PII in each) | 9,064 | 0 | 305 req/s | 240 ms | 590 ms | 810 ms |
| Cache hit (identical prompts) | 14,574 | 0 | 488 req/s | 150 ms | 270 ms | 340 ms |

Single worker, no middlewares: ~146 req/s (the 100 users saturate one Python process; latency is queueing, not the 20 ms upstream). Retries and fallbacks absorbed every injected failure: in one run 6,965 retries and 2,997 fallbacks produced zero client errors. Reproduce:

```bash
GATEWAY_WORKERS=4 GATEWAY_ENABLED_PROVIDERS='["fake","fake2"]' GATEWAY_ROUTES_FILE=config/routes.loadtest.json \
GATEWAY_FAKE_FAILURE_RATE=0.2 GATEWAY_FAKE_LATENCY_MS=20 docker compose up -d --build
LOCUST_UNIQUE=1 LOCUST_API_KEY=sk-gw-... uv run locust -f loadtest/locustfile.py --headless -u 100 -r 50 -t 30s --host http://localhost:8000
```

Caveat: these numbers say nothing about real providers; real LLM latency (seconds) dominates, so the gateway's own overhead is the thing measured here.

## Docs

- [ARCHITECTURE.md](ARCHITECTURE.md): request flow diagram and components
- [docs/adr/](docs/adr): design decisions
- [CONTRACT.md](CONTRACT.md): middleware interface
- PII: [gateway/middleware/pii/README.md](gateway/middleware/pii/README.md), cache: [gateway/middleware/cache/README.md](gateway/middleware/cache/README.md), [docs/cache.md](docs/cache.md)

## Not covered / known limits

- Live calls to Anthropic, OpenAI and OpenRouter are not exercised (paid accounts); adapters are tested against mocked HTTP. Ollama is likewise tested against a mock, not a running server.
- Rate limiting is a fixed window; a client can burst 2x at a window boundary.
- Streaming: retries/fallback only happen before the first chunk; a mid-stream upstream failure ends the stream with an SSE `error` event. Only choice 0 of `n>1` streams gets PII restoration.
- Budgets are checked before and charged after a request, so concurrent requests can overshoot a budget slightly.
- Only chat completions (no embeddings, images, tool-call translation for Anthropic).
