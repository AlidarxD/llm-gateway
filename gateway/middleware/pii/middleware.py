"""PII redaction middleware: redact / block / log-only, per API key."""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from gateway.middleware.pii.detectors import Detector, default_detectors, find_all
from gateway.schemas import ChatChoice, ChatMessage, ChatRequest, ChatResponse, RequestContext

logger = logging.getLogger("gateway.pii")

VAULT_KEY = "_pii_vault"
_PLACEHOLDER_RE = re.compile(r"<([A-Z]+)_(\d+)>")


class PIIMode(StrEnum):
    REDACT = "redact"
    BLOCK = "block"
    LOG_ONLY = "log_only"


@dataclass(slots=True)
class PIIConfig:
    default_mode: PIIMode = PIIMode.REDACT
    per_key: Mapping[str, PIIMode] = field(default_factory=dict)
    # Entities to look for; None = every entity the detectors produce.
    entities: frozenset[str] | None = None

    def mode_for(self, api_key_id: str) -> PIIMode:
        return self.per_key.get(api_key_id, self.default_mode)


class PIIVault:
    """Placeholder <-> original mapping. Lives in RequestContext.extra only.

    repr/str never show values, so an accidental `log(ctx)` cannot leak PII.
    """

    __slots__ = ("_by_placeholder", "_by_value", "_counters")

    def __init__(self) -> None:
        self._by_placeholder: dict[str, str] = {}
        self._by_value: dict[tuple[str, str], str] = {}
        self._counters: Counter[str] = Counter()

    def placeholder(self, entity: str, value: str) -> str:
        key = (entity, value)
        existing = self._by_value.get(key)
        if existing is not None:
            return existing
        self._counters[entity] += 1
        ph = f"<{entity}_{self._counters[entity]}>"
        self._by_value[key] = ph
        self._by_placeholder[ph] = value
        return ph

    def original(self, placeholder: str) -> str | None:
        return self._by_placeholder.get(placeholder)

    def restore(self, text: str) -> str:
        if not self._by_placeholder or "<" not in text:
            return text
        return _PLACEHOLDER_RE.sub(lambda m: self._by_placeholder.get(m.group(0), m.group(0)), text)

    def __len__(self) -> int:
        return len(self._by_placeholder)

    def __repr__(self) -> str:
        return f"<PIIVault entries={len(self)} [values hidden]>"

    __str__ = __repr__

    def __reduce__(self) -> Any:
        raise TypeError("PIIVault must not be serialized")


class StreamRestorer:
    """Restores placeholders in a stream of text deltas, even when split across chunks."""

    _PARTIAL_RE = re.compile(r"<[A-Z]*(?:_\d*)?$")
    _MAX_PARTIAL = 24

    def __init__(self, vault: PIIVault) -> None:
        self._vault = vault
        self._buf = ""

    def feed(self, delta: str) -> str:
        self._buf += delta
        cut = len(self._buf)
        lt = self._buf.rfind("<")
        if lt != -1 and len(self._buf) - lt <= self._MAX_PARTIAL:
            if self._PARTIAL_RE.match(self._buf, lt):
                cut = lt
        out, self._buf = self._buf[:cut], self._buf[cut:]
        return self._vault.restore(out)

    def flush(self) -> str:
        out, self._buf = self._buf, ""
        return self._vault.restore(out)


def get_vault(ctx: RequestContext) -> PIIVault | None:
    v = ctx.extra.get(VAULT_KEY)
    return v if isinstance(v, PIIVault) else None


class PIIMiddleware:
    """`before`: redact/block/log request text. `after`: restore originals in the response."""

    name = "pii"

    def __init__(
        self,
        config: PIIConfig | None = None,
        detectors: Iterable[Detector] | None = None,
        on_detect: Callable[[str, Mapping[str, int]], None] | None = None,
    ) -> None:
        self.config = config or PIIConfig()
        self.detectors: list[Detector] = list(detectors) if detectors else default_detectors()
        self._on_detect = on_detect

    # ------------------------------------------------------------------ helpers
    def _active(self) -> list[Detector]:
        ents = self.config.entities
        return self.detectors if ents is None else [d for d in self.detectors if d.entity in ents]

    def redact_text(self, text: str, vault: PIIVault | None) -> tuple[str, Counter[str]]:
        """Return (text with placeholders, entity counts). vault=None means count only."""
        matches = find_all(text, self._active())
        counts: Counter[str] = Counter(m.entity for m in matches)
        if not matches or vault is None:
            return text, counts
        out: list[str] = []
        pos = 0
        for m in matches:
            out.append(text[pos : m.start])
            out.append(vault.placeholder(m.entity, m.value))
            pos = m.end
        out.append(text[pos:])
        return "".join(out), counts

    def _map_message(self, msg: ChatMessage, fn: Callable[[str], str]) -> ChatMessage:
        updates: dict[str, Any] = {}
        if isinstance(msg.content, str):
            updates["content"] = fn(msg.content)
        elif isinstance(msg.content, list):
            parts: list[dict[str, Any]] = []
            for part in msg.content:
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    part = {**part, "text": fn(part["text"])}
                parts.append(part)
            updates["content"] = parts
        if msg.tool_calls:
            calls: list[dict[str, Any]] = []
            for call in msg.tool_calls:
                fnc = call.get("function")
                if isinstance(fnc, dict) and isinstance(fnc.get("arguments"), str):
                    call = {**call, "function": {**fnc, "arguments": fn(fnc["arguments"])}}
                calls.append(call)
            updates["tool_calls"] = calls
        return msg.model_copy(update=updates) if updates else msg

    # ------------------------------------------------------------------ Middleware API
    async def before(self, req: ChatRequest, ctx: RequestContext) -> ChatRequest | ChatResponse:
        mode = self.config.mode_for(ctx.api_key_id)
        started = time.perf_counter()

        if mode is PIIMode.REDACT:
            vault = PIIVault()
            counts: Counter[str] = Counter()

            def redact(text: str) -> str:
                new, c = self.redact_text(text, vault)
                counts.update(c)
                return new

            new_msgs = [self._map_message(m, redact) for m in req.messages]
            if counts:
                ctx.extra[VAULT_KEY] = vault
            self._report(ctx, mode, counts, started)
            return req.model_copy(update={"messages": new_msgs}) if counts else req

        counts = Counter()

        def count_only(text: str) -> str:
            _, c = self.redact_text(text, None)
            counts.update(c)
            return text

        for m in req.messages:
            self._map_message(m, count_only)
        self._report(ctx, mode, counts, started)

        if mode is PIIMode.BLOCK and counts:
            ctx.extra["pii_blocked"] = True
            kinds = ", ".join(sorted(counts))
            return ChatResponse(
                id=f"gw-pii-{ctx.request_id}",
                created=int(time.time()),
                model=req.model,
                choices=[
                    ChatChoice(
                        message=ChatMessage(
                            role="assistant",
                            content=f"Request blocked: it contains personal data ({kinds}).",
                        ),
                        finish_reason="content_filter",
                    )
                ],
            )
        return req

    async def after(
        self, req: ChatRequest, resp: ChatResponse, ctx: RequestContext
    ) -> ChatResponse:
        vault = get_vault(ctx)
        if vault is None or len(vault) == 0:
            return resp
        choices = [
            ChatChoice(
                index=c.index,
                message=self._map_message(c.message, vault.restore),
                finish_reason=c.finish_reason,
            )
            for c in resp.choices
        ]
        return resp.model_copy(update={"choices": choices})

    def stream_restorer(self, ctx: RequestContext) -> StreamRestorer | None:
        """For the core: restore placeholders across streamed deltas of one choice."""
        vault = get_vault(ctx)
        return StreamRestorer(vault) if vault is not None else None

    def _report(
        self, ctx: RequestContext, mode: PIIMode, counts: Counter[str], started: float
    ) -> None:
        ctx.extra["pii_detected"] = dict(counts)  # counts only, never values
        ctx.extra["pii_mode"] = mode.value
        if counts:
            logger.info(
                "pii request_id=%s mode=%s entities=%s", ctx.request_id, mode.value, dict(counts)
            )
            if self._on_detect is not None:
                self._on_detect(mode.value, dict(counts))
        ctx.extra["pii_latency_ms"] = (time.perf_counter() - started) * 1000
