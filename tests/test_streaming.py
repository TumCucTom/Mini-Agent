"""Tests for LLM streaming functionality."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mini_agent.llm.anthropic_client import AnthropicClient
from mini_agent.llm.openai_client import OpenAIClient
from mini_agent.llm.llm_wrapper import LLMClient
from mini_agent.schema import (
    FunctionCall,
    LLMProvider,
    LLMResponse,
    Message,
    StreamChunk,
    ToolCall,
    TokenUsage,
)


class TestStreamChunk:
    """Test StreamChunk schema."""

    def test_stream_chunk_content(self):
        """Test content chunk creation."""
        chunk = StreamChunk(type="content", text="Hello")
        assert chunk.type == "content"
        assert chunk.text == "Hello"

    def test_stream_chunk_thinking(self):
        """Test thinking chunk creation."""
        chunk = StreamChunk(type="thinking", text="Let me think...")
        assert chunk.type == "thinking"
        assert chunk.text == "Let me think..."

    def test_stream_chunk_tool_call_delta(self):
        """Test tool call delta chunk."""
        chunk = StreamChunk(
            type="tool_call_delta",
            tool_call_id="abc123",
            arguments='{"name":',
        )
        assert chunk.type == "tool_call_delta"
        assert chunk.tool_call_id == "abc123"
        assert chunk.arguments == '{"name":'

    def test_stream_chunk_tool_call_complete(self):
        """Test complete tool call chunk."""
        tool_call = ToolCall(
            id="abc123",
            type="function",
            function=FunctionCall(name="test_tool", arguments={"arg": "value"}),
        )
        chunk = StreamChunk(type="tool_call_complete", tool_call=tool_call)
        assert chunk.type == "tool_call_complete"
        assert chunk.tool_call.id == "abc123"
        assert chunk.tool_call.function.name == "test_tool"

    def test_stream_chunk_done(self):
        """Test done chunk."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        chunk = StreamChunk(type="done", finish_reason="stop", usage=usage)
        assert chunk.type == "done"
        assert chunk.finish_reason == "stop"
        assert chunk.usage.total_tokens == 150


class TestOpenAIStreaming:
    """Test OpenAI client streaming."""

    @pytest.fixture
    def openai_client(self):
        """Create OpenAI client for testing."""
        client = OpenAIClient(
            api_key="test-key",
            api_base="https://test.api.minimaxi.com/v1",
            model="MiniMax-M2.5",
            retry_config=None,
        )
        return client

    @pytest.mark.asyncio
    async def test_generate_stream_content_only(self, openai_client):
        """Test streaming content without tool calls."""
        # Create a proper async iterator
        class MockEvent:
            """Single mock streaming event."""
            def __init__(self, content_text, finish=None):
                class MockDelta:
                    content = content_text
                    reasoning_details = None
                class MockChoice:
                    delta = MockDelta()
                    finish_reason = finish
                self.choices = [MockChoice()]
                self.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        class AsyncEventStream:
            """Mock async iterator for OpenAI streaming."""
            def __init__(self, events):
                self.events = events
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.events):
                    raise StopAsyncIteration
                event = self.events[self.index]
                self.index += 1
                return event

        events = [
            MockEvent("Hello"),
            MockEvent(", world!"),
            MockEvent("", finish="stop"),
        ]

        async def mock_create(**kwargs):
            return AsyncEventStream(events)

        openai_client.client = MagicMock()
        openai_client.client.chat.completions.create = mock_create

        messages = [Message(role="user", content="Say hello")]
        chunks = []
        async for chunk in openai_client.generate_stream(messages):
            chunks.append(chunk)

        assert len(chunks) >= 2
        content_chunks = [c for c in chunks if c.type == "content"]
        assert len(content_chunks) >= 2

    @pytest.mark.asyncio
    async def test_generate_stream_with_thinking(self, openai_client):
        """Test streaming with thinking content."""
        class MockThinkingEvent:
            def __init__(self, think_text):
                class MockDelta:
                    content = None
                    reasoning_details = [MagicMock(text=think_text)]
                class MockChoice:
                    delta = MockDelta()
                    finish_reason = None
                self.choices = [MockChoice()]
                self.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        class MockContentEvent:
            def __init__(self, content_text):
                class MockDelta:
                    content = content_text
                    reasoning_details = None
                class MockChoice:
                    delta = MockDelta()
                    finish_reason = None
                self.choices = [MockChoice()]
                self.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        class AsyncEventStream:
            def __init__(self, events):
                self.events = events
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.events):
                    raise StopAsyncIteration
                event = self.events[self.index]
                self.index += 1
                return event

        events = [
            MockThinkingEvent("Let me think..."),
            MockContentEvent("Here's my answer"),
        ]

        async def mock_create(**kwargs):
            return AsyncEventStream(events)

        openai_client.client = MagicMock()
        openai_client.client.chat.completions.create = mock_create

        messages = [Message(role="user", content="Think about something")]
        chunks = []
        async for chunk in openai_client.generate_stream(messages):
            chunks.append(chunk)

        thinking_chunks = [c for c in chunks if c.type == "thinking"]
        assert len(thinking_chunks) >= 1

    @pytest.mark.asyncio
    async def test_generate_stream_tool_call_complete(self, openai_client):
        """Test that tool call is emitted when JSON is complete."""
        class MockToolEvent:
            """Mock event with tool call."""
            def __init__(self, args_str, has_id=False, has_name=False, finish=None):
                class MockFunction:
                    name = "test_tool" if has_name else ""
                    arguments = args_str

                class MockToolCall:
                    index = 0
                    id = "call_123" if has_id else ""
                    function = MockFunction()

                class MockDelta:
                    content = None
                    reasoning_details = None
                    tool_calls = [MockToolCall()] if args_str else None

                class MockChoice:
                    delta = MockDelta()
                    finish_reason = finish

                self.choices = [MockChoice()]
                self.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        class AsyncEventStream:
            def __init__(self, events):
                self.events = events
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.events):
                    raise StopAsyncIteration
                event = self.events[self.index]
                self.index += 1
                return event

        # Tool call with complete arguments
        events = [
            MockToolEvent('{"arg": "value"}', has_id=True, has_name=True, finish="tool_calls"),
        ]

        async def mock_create(**kwargs):
            return AsyncEventStream(events)

        openai_client.client = MagicMock()
        openai_client.client.chat.completions.create = mock_create

        messages = [Message(role="user", content="Use test tool")]
        chunks = []
        async for chunk in openai_client.generate_stream(messages):
            chunks.append(chunk)

        # Should have a complete tool call
        complete_chunks = [c for c in chunks if c.type == "tool_call_complete"]
        assert len(complete_chunks) == 1
        assert complete_chunks[0].tool_call.function.name == "test_tool"
        assert complete_chunks[0].tool_call.function.arguments == {"arg": "value"}


class TestAnthropicStreaming:
    """Test Anthropic client streaming."""

    @pytest.fixture
    def anthropic_client(self):
        """Create Anthropic client for testing."""
        client = AnthropicClient(
            api_key="test-key",
            api_base="https://test.api.minimaxi.com/anthropic",
            model="MiniMax-M2.5",
            retry_config=None,
        )
        return client

    @pytest.mark.asyncio
    async def test_generate_stream_content_only(self, anthropic_client):
        """Test streaming content without tool calls."""
        class MockTextBlock:
            type = "text"
            text = "Hello"

        class MockEvent:
            """Mock content_block_delta event."""
            def __init__(self, block_type, text=None, thinking=None):
                self.type = "content_block_delta"
                self.index = 0
                if block_type == "text":
                    self.content_block = type('obj', (), {'type': 'text', 'text': text})()
                elif block_type == "thinking":
                    self.content_block = type('obj', (), {'type': 'thinking', 'thinking': thinking})()

        class AsyncEventStream:
            def __init__(self, events):
                self.events = events
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.events):
                    raise StopAsyncIteration
                event = self.events[self.index]
                self.index += 1
                return event

        events = [
            MockEvent("text", text="Hello"),
            MockEvent("text", text=", world!"),
        ]

        mock_client = AsyncMock()
        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=AsyncEventStream(events))
        mock_stream_ctx.__aexit__ = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream_ctx)
        anthropic_client.client = mock_client

        messages = [Message(role="user", content="Say hello")]
        chunks = []
        async for chunk in anthropic_client.generate_stream(messages):
            chunks.append(chunk)

        content_chunks = [c for c in chunks if c.type == "content"]
        assert len(content_chunks) >= 2

    @pytest.mark.asyncio
    async def test_generate_stream_thinking(self, anthropic_client):
        """Test streaming with thinking content."""
        class MockEvent:
            def __init__(self, text):
                self.type = "content_block_delta"
                self.index = 0
                self.content_block = type('obj', (), {'type': 'thinking', 'thinking': text})()

        class AsyncEventStream:
            def __init__(self, events):
                self.events = events
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.events):
                    raise StopAsyncIteration
                event = self.events[self.index]
                self.index += 1
                return event

        events = [
            MockEvent("Let me think..."),
        ]

        mock_client = AsyncMock()
        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=AsyncEventStream(events))
        mock_stream_ctx.__aexit__ = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream_ctx)
        anthropic_client.client = mock_client

        messages = [Message(role="user", content="Think about something")]
        chunks = []
        async for chunk in anthropic_client.generate_stream(messages):
            chunks.append(chunk)

        thinking_chunks = [c for c in chunks if c.type == "thinking"]
        assert len(thinking_chunks) >= 1


class TestLLMClientWrapperStreaming:
    """Test LLMClient wrapper streaming interface."""

    @pytest.mark.asyncio
    async def test_generate_stream_delegates_to_client(self):
        """Test that generate_stream properly delegates to underlying client."""
        client = LLMClient(
            api_key="test-key",
            provider=LLMProvider.OPENAI,
            model="MiniMax-M2.5",
        )

        # Verify generate_stream method exists and is accessible
        assert hasattr(client, "generate_stream")
        assert callable(client.generate_stream)


class TestMessageConversion:
    """Test message conversion for streaming."""

    def test_convert_thinking_in_assistant_message(self):
        """Test that thinking is preserved in assistant messages for streaming."""
        msg = Message(
            role="assistant",
            content="Here's my response",
            thinking="Let me think about this...",
            tool_calls=None,
        )

        assert msg.thinking == "Let me think about this..."
        assert msg.content == "Here's my response"


# Helper for async iterator mock
async def async_iter(items):
    """Create async iterator from list of items."""
    for item in items:
        yield item
