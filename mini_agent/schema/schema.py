from enum import Enum
from typing import Any

from pydantic import BaseModel


class LLMProvider(str, Enum):
    """LLM provider types."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class FunctionCall(BaseModel):
    """Function call details."""

    name: str
    arguments: dict[str, Any]  # Function arguments as dict


class ToolCall(BaseModel):
    """Tool call structure."""

    id: str
    type: str  # "function"
    function: FunctionCall


class Message(BaseModel):
    """Chat message."""

    role: str  # "system", "user", "assistant", "tool"
    content: str | list[dict[str, Any]]  # Can be string or list of content blocks
    thinking: str | None = None  # Extended thinking content for assistant messages
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None  # For tool role


class TokenUsage(BaseModel):
    """Token usage statistics from LLM API response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """LLM response."""

    content: str
    thinking: str | None = None  # Extended thinking blocks
    tool_calls: list[ToolCall] | None = None
    finish_reason: str
    usage: TokenUsage | None = None  # Token usage from API response


class StreamChunk(BaseModel):
    """A single chunk from a streaming LLM response."""

    type: str  # "thinking" | "content" | "tool_call_start" | "tool_call_delta" | "tool_call_complete" | "done"
    text: str | None = None  # For thinking / content chunks
    tool_call_id: str | None = None  # For tool call chunks
    tool_name: str | None = None  # For tool_call_start
    arguments: str | None = None  # For tool_call_delta (partial JSON string fragment)
    tool_call: ToolCall | None = None  # For tool_call_complete (full tool call)
    finish_reason: str | None = None  # For done
    usage: TokenUsage | None = None  # For done
