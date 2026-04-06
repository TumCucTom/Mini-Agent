"""SubAgent Tool - Spawn a child agent to handle a subtask independently."""

import asyncio
from pathlib import Path
from typing import Any, Callable, Optional

from .base import Tool, ToolResult


class SubAgentTool(Tool):
    """Tool for spawning a child agent to handle a subtask.

    The child agent operates in an isolated message history and can use
    a subset of the parent's tools. Results are returned as a ToolResult
    once the child completes.

    Use this when a task can be broken into independent parts that benefit
    from separate reasoning chains, or when you want to delegate a complex
    subtask to a fresh agent context.

    Example:
        subagent(
            task="Research the history of the Roman Empire",
            tool_names=["bash", "read", "write"],
            max_steps=30
        )
    """

    def __init__(
        self,
        llm_client: "LLMClient",
        system_prompt: str = "You are a helpful assistant focused on completing tasks thoroughly.",
        default_max_steps: int = 20,
        workspace_dir: str = "./workspace",
    ):
        """Initialize SubAgentTool.

        Args:
            llm_client: LLMClient instance to use for the child agent.
            system_prompt: Default system prompt for child agents.
            default_max_steps: Default max steps for child agents if not specified.
            workspace_dir: Workspace directory for child agents.
        """
        self._llm = llm_client
        self._system_prompt = system_prompt
        self._default_max_steps = default_max_steps
        self._workspace_dir = workspace_dir
        # The agent reference is set after the tool is registered with an agent.
        # This uses a MutableContainer so the reference can be injected post-creation.
        self._agent_ref: Optional["Agent"] = None

    def bind_agent(self, agent: "Agent") -> None:
        """Bind this tool to an agent instance.

        Must be called after the agent is created and before the tool is used.
        Allows the tool to access the agent's current tool set at execute time.
        """
        self._agent_ref = agent

    @property
    def name(self) -> str:
        return "subagent"

    @property
    def description(self) -> str:
        return (
            "Spawn a child agent to handle a subtask independently. "
            "Use when a task can be broken into parallel or independent parts, "
            "or when you want a fresh reasoning context for a complex subtask. "
            "The child agent has its own isolated message history and optionally "
            "a restricted set of tools. Returns the child's final response."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "The task, question, or instruction to give to the child agent. "
                        "Be specific about what you need."
                    ),
                },
                "system_prompt": {
                    "type": "string",
                    "description": (
                        "Optional system prompt override for this child agent. "
                        "Use this to give the child agent specific instructions or context."
                    ),
                },
                "tool_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of tool names to delegate to the child agent. "
                        "If not provided, the child agent has no external tools. "
                        "Available tools in the parent: bash, read, write, edit, "
                        "session_note, recall_notes, and any loaded MCP/skill tools."
                    ),
                },
                "max_steps": {
                    "type": "integer",
                    "description": (
                        f"Maximum steps for the child agent (default: {self._default_max_steps}). "
                        "Increase for complex tasks that need more reasoning steps."
                    ),
                },
            },
            "required": ["task"],
        }

    async def execute(
        self,
        task: str,
        system_prompt: Optional[str] = None,
        tool_names: Optional[list[str]] = None,
        max_steps: Optional[int] = None,
    ) -> ToolResult:
        from mini_agent.agent import Agent

        if self._agent_ref is None:
            return ToolResult(
                success=False,
                content="",
                error="SubAgentTool: agent not bound. Call bind_agent(agent) first.",
            )

        # Resolve tools from the parent's current tool set
        available = list(self._agent_ref.tools.values())
        available_names = {t.name for t in available}

        if tool_names is not None:
            child_tool_list = [t for t in available if t.name in tool_names]
            missing = set(tool_names) - available_names
            if missing:
                return ToolResult(
                    success=False,
                    content="",
                    error=f"Tool(s) not available in parent agent: {', '.join(sorted(missing))}. "
                          f"Available: {', '.join(sorted(available_names))}",
                )
        else:
            child_tool_list = []

        # Build system prompt
        prompt = system_prompt or self._system_prompt
        if "Current Workspace" not in prompt:
            workspace = Path(self._workspace_dir).resolve()
            prompt = (
                f"{prompt}\n\n"
                f"## Current Workspace\n"
                f"You are working in: `{workspace}`\n"
                f"All relative paths are resolved relative to this directory."
            )

        # Create child agent with isolated history
        child = Agent(
            llm_client=self._llm,
            system_prompt=prompt,
            tools=child_tool_list,
            max_steps=max_steps or self._default_max_steps,
            workspace_dir=str(Path(self._workspace_dir).resolve()),
            stream=False,
        )
        child.add_user_message(task)

        # Run child agent
        try:
            result = await child.run()
            return ToolResult(success=True, content=result)
        except Exception as exc:
            return ToolResult(success=False, content="", error=f"SubAgent error: {exc}")
