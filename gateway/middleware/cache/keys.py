"""Request normalization, hashing and cacheability rules."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from gateway.schemas import ChatMessage, ChatRequest, ChatResponse

_WS = re.compile(r"\s+")
_PLACEHOLDER = re.compile(r"<[A-Z]+_\d+>")

# Request fields that change the answer; everything else (stream, user, ...) is ignored.
_PARAM_FIELDS = (
    "temperature",
    "top_p",
    "n",
    "stop",
    "max_tokens",
    "presence_penalty",
    "frequency_penalty",
)


def normalize_text(text: str) -> str:
    return _WS.sub(" ", text).strip()


def message_text(msg: ChatMessage) -> str:
    if isinstance(msg.content, str):
        return normalize_text(msg.content)
    if isinstance(msg.content, list):
        return normalize_text(
            " ".join(p.get("text", "") for p in msg.content if p.get("type") == "text")
        )
    return ""


def _sha(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def params_of(req: ChatRequest) -> dict[str, Any]:
    return {f: getattr(req, f) for f in _PARAM_FIELDS if getattr(req, f) is not None}


def exact_key(req: ChatRequest) -> str:
    """Hash of the normalized request (model, params, role+text of every message)."""
    return _sha(
        {
            "model": req.model,
            "params": params_of(req),
            "messages": [[m.role, message_text(m)] for m in req.messages],
        }
    )


def split_for_semantic(req: ChatRequest) -> tuple[str, str] | None:
    """(context_hash, last_user_text) or None if the request has no plain final user turn.

    The semantic match compares only the last user message; everything before it
    (system prompt, history) and all sampling params must be identical.
    """
    if not req.messages or req.messages[-1].role != "user":
        return None
    last = message_text(req.messages[-1])
    if not last:
        return None
    ctx = _sha(
        {
            "model": req.model,
            "params": params_of(req),
            "history": [[m.role, message_text(m)] for m in req.messages[:-1]],
        }
    )
    return ctx, last


def has_placeholders(text: str) -> bool:
    return bool(_PLACEHOLDER.search(text))


def request_cacheable(req: ChatRequest, max_temperature: float) -> bool:
    if req.tools or req.tool_choice:
        return False
    if req.temperature is not None and req.temperature > max_temperature:
        return False
    if req.n is not None and req.n > 1:
        return False
    for m in req.messages:
        if m.tool_calls or m.tool_call_id or m.role == "tool":
            return False
    return True


def response_cacheable(resp: ChatResponse) -> bool:
    if len(resp.choices) != 1:
        return False
    c = resp.choices[0]
    return c.finish_reason == "stop" and not c.message.tool_calls and bool(c.message.content)
