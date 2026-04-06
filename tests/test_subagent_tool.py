"""Test cases for SubAgentTool."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from mini_agent.tools.subagent_tool import SubAgentTool


@pytest.mark.asyncio
async def test_subagent_tool_basic_properties():
    """Test SubAgentTool name, description, parameters."""
    tool = SubAgentTool(
        llm_client=MagicMock(),
        workspace_dir="/tmp",
    )
    assert tool.name == "subagent"
    assert "spawn" in tool.description.lower()
    assert "task" in tool.parameters["properties"]
    assert tool.parameters["required"] == ["task"]


@pytest.mark.asyncio
async def test_subagent_tool_unbound_agent_error():
    """Test that execute fails cleanly when agent is not bound."""
    tool = SubAgentTool(
        llm_client=MagicMock(),
        workspace_dir="/tmp",
    )
    result = await tool.execute(task="hello")
    assert not result.success
    assert "not bound" in result.error


@pytest.mark.asyncio
async def test_subagent_tool_missing_tools_error():
    """Test error when requested tool is not in parent agent's tools."""
    mock_llm = MagicMock()
    tool = SubAgentTool(
        llm_client=mock_llm,
        workspace_dir="/tmp",
    )

    # Mock agent with some tools
    mock_tool = MagicMock()
    mock_tool.name = "bash"
    mock_agent = MagicMock()
    mock_agent.tools = {"bash": mock_tool}
    tool.bind_agent(mock_agent)

    # Try to delegate a non-existent tool
    result = await tool.execute(
        task="test task",
        tool_names=["nonexistent_tool"],
    )
    assert not result.success
    assert "not available" in result.error


@pytest.mark.asyncio
async def test_subagent_tool_delegates_valid_tools():
    """Test that valid tool_names filter works correctly."""
    mock_llm = MagicMock()
    tool = SubAgentTool(
        llm_client=mock_llm,
        workspace_dir="/tmp",
    )

    # Create mock tools
    bash_tool = MagicMock()
    bash_tool.name = "bash"
    read_tool = MagicMock()
    read_tool.name = "read"

    mock_agent = MagicMock()
    mock_agent.tools = {"bash": bash_tool, "read": read_tool}
    tool.bind_agent(mock_agent)

    # This would create child with only bash - we can't fully test execute without
    # a real LLM, but we can verify the filter logic ran (child agent was created)
    result = await tool.execute(
        task="test",
        tool_names=["bash"],  # valid - filter passed
    )
    # The mock LLM can't make real calls, so child.run() fails internally.
    # But we can confirm: no "not available" error → tool delegation worked
    assert "not available" not in (result.error or result.content or "")
