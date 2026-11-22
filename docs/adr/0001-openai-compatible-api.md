# 1. Expose the OpenAI chat-completions API

**Status:** accepted

**Context.** The gateway must be adoptable without code changes. Most SDKs and tools speak the OpenAI chat-completions protocol.

**Decision.** Request/response/streaming schemas mirror OpenAI (`extra="allow"` keeps unknown fields). Errors use the `{"error": {message, type, code}}` envelope. Other providers are translated *to* this schema in their adapters (Anthropic Messages API → chat completions).

**Consequences.** Drop-in for existing clients. Provider-specific features that have no OpenAI equivalent (Anthropic tool-use blocks, thinking) are not exposed yet.
