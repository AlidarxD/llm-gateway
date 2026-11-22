"""OpenAI chat-completions compatible schemas plus gateway request context."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool", "developer"]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Role
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    n: int | None = Field(default=None, ge=1)
    stream: bool = False
    stop: str | list[str] | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage = Field(default_factory=Usage)


class ChatDelta(BaseModel):
    role: Role | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatChunkChoice(BaseModel):
    index: int = 0
    delta: ChatDelta
    finish_reason: str | None = None


class ChatChunk(BaseModel):
    """Streaming chunk (`chat.completion.chunk`), sent as SSE `data:` lines."""

    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatChunkChoice]
    usage: Usage | None = None


class BudgetInfo(BaseModel):
    """Remaining budget for the calling key; None means unlimited."""

    tokens_remaining: int | None = None
    usd_remaining: Decimal | None = None


class RequestContext(BaseModel):
    """Per-request state shared by all middlewares. `extra` is a scratchpad for middlewares."""

    api_key_id: str
    tenant_id: str
    request_id: str
    budget: BudgetInfo = Field(default_factory=BudgetInfo)
    extra: dict[str, Any] = Field(default_factory=dict)
