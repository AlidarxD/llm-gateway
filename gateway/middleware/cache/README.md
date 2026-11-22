# Cache middleware

`CacheMiddleware(redis, embedder=None, config=CacheConfig(), metrics=None)` sits after PII (`pii -> cache -> provider`),
so only redacted text is hashed, embedded and stored.

- **before**: exact lookup (sha256 of model + sampling params + role/text of all messages, whitespace-normalized),
  then (if enabled) semantic lookup. A hit returns the stored `ChatResponse` (short-circuit).
- **after**: stores the provider response (exact + semantic) with the TTL.
- Never cached / never served: `temperature > max_temperature` (default 0.3), `tools`/`tool_choice`, tool messages or
  tool calls, `n > 1`, responses whose `finish_reason != "stop"` or that contain tool calls.
- Tenant isolation: `tenant_id` and model are part of every Redis key (`gw:cache:x:{tenant}:{hash}`,
  `gw:cache:s:{tenant}:{model}:{id}`); other tenants' entries are never loaded. Covered by tests.
- Failures (Redis down, embedder error) degrade to a miss; requests never fail because of the cache.

## Semantic cache

Embeddings: `intfloat/multilingual-e5-small` ONNX on CPU (`E5OnnxEmbedder`, fp32, downloaded from the HF hub on first use).
Index: fallback implementation without RediSearch: JSON entries (vector + response) in Redis, ids in a sorted set capped at
`max_semantic_entries`, brute-force cosine in numpy. (A RediSearch backend can implement the same `search/add` interface.)

A semantic hit requires: same tenant + model alias, **identical history and sampling params** (only the last user message
may differ), similarity >= threshold (`CacheConfig.thresholds[model_alias]`, default `0.94`), and a matching **lexical
guard** (same numbers, same negation flag). Semantic reuse is skipped when the text contains PII placeholders
(`<IIN_1>`); the exact cache still works for those.

**It is opt-in (`semantic_enabled=False` by default)** because of the evaluation below.

## Metrics hooks

Pass any object implementing `CacheMetrics` (`on_lookup(kind, outcome, latency_seconds)`, `on_store(kind, latency_seconds)`;
`kind` in `exact|semantic`, `outcome` in `hit|miss|skip|error`) - e.g. one that increments Prometheus counters/histograms.
`InMemoryCacheMetrics` is a reference implementation. Per request, `ctx.extra` also holds `cache_hit` (bool),
`cache_type` (`exact|semantic|None`) and `cache_similarity` (semantic hits only).

## Evaluation (summary; details and curve in [docs/cache.md](../../../docs/cache.md))

- Threshold chosen from data: **0.94** (lowest threshold with <= 0.5% false hits into an unrelated group).
- Paraphrase hit rate there: 34% (RU 24%, KK 51%, EN 55%, cross-lingual 3%).
- Hard-negative false-hit rate (minimal pairs: different number/entity/negation): **53%** with the guard, 64% without.
  No threshold gives <= 2% on those pairs with a usable hit rate. Use semantic caching only for low-stakes aliases.
- Embedding latency (CPU, 2 ONNX threads, 12-core x86_64): single text p50 22 ms, p95 45 ms; batch of 8 ~150 ms.
