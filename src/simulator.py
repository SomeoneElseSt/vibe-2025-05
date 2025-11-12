"""Simulation module for fixing failed agent criteria.

This module provides primitives for analyzing failed judgments and generating
modifications to agent prompts and tools.
"""

import asyncio
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from dedalus_labs import AsyncDedalus

from .types import (
    JudgmentResult,
    Conversation,
    FixerTask,
    AgentModification,
    SimulationResult
)
from .config import (
    FIXER_MODEL,
    FIXER_TEMPERATURE,
    AVAILABLE_MCP_SERVERS,
    TOOL_ADDING_INSTRUCTIONS
)

# Load environment variables
load_dotenv()


# Pydantic models for structured output
class ToolDefinition(BaseModel):
    """Definition of a new tool to add."""
    name: str
    description: str
    purpose: str


class MCPServerAddition(BaseModel):
    """MCP server to add."""
    server_name: str
    reason: str


class PromptModification(BaseModel):
    """Structured modification response from fixer agent."""
    modified_prompt: str = Field(description="The complete modified agent prompt")
    changes_made: List[str] = Field(description="List of specific changes made to the prompt")
    reasoning: str = Field(description="Explanation of why these changes will fix the issue")
    tools_to_add: Optional[List[ToolDefinition]] = Field(default=None, description="New tools to add, if any")
    mcp_servers_to_add: Optional[List[MCPServerAddition]] = Field(default=None, description="MCP servers to add, if any")


def format_conversation_for_fixer(conversation: Conversation) -> str:
    """Format a conversation for the fixer agent to analyze.

    Args:
        conversation: Conversation to format

    Returns:
        Formatted string representation
    """
    lines = []
    lines.append("=== FAILED CONVERSATION ===")

    for i, message in enumerate(conversation["messages"]):
        role = message["role"]
        content = message["content"]
        lines.append(f"\nTurn {i+1} - {role}:")
        lines.append(content)

    lines.append("\n=== END CONVERSATION ===")
    return "\n".join(lines)


async def fix_single_issue(
    task: FixerTask,
    base_agent_prompt: str,
    model: str = FIXER_MODEL,
    temperature: float = FIXER_TEMPERATURE
) -> Optional[AgentModification]:
    """Fix a single failed criterion using a fixer agent.

    Args:
        task: Information about the failed criterion
        base_agent_prompt: Current agent prompt that needs fixing
        model: Model identifier to use for fixer agent
        temperature: Sampling temperature

    Returns:
        AgentModification or None if failed
    """
    client = AsyncDedalus()

    # Format the conversation
    conversation_text = format_conversation_for_fixer(task["conversation"])

    # Build the fixer prompt
    fixer_system_prompt = f"""You are an expert AI agent architect. Your job is to analyze failed agent conversations and improve the agent's system prompt or suggest tools/MCP servers to add.

You will be given:
1. The current agent prompt
2. A criterion that the agent failed to meet
3. The reasoning for why it failed
4. The actual conversation that failed

Your task is to modify the agent's prompt to fix the issue. You can:
- Rewrite or enhance parts of the prompt
- Add new instructions
- Make the prompt more specific
- Suggest adding custom tools (Python functions)
- Suggest adding MCP servers for external capabilities

{AVAILABLE_MCP_SERVERS}

{TOOL_ADDING_INSTRUCTIONS}

Be specific and actionable in your modifications."""

    user_prompt = f"""Analyze this failed agent conversation and fix the issue.

CURRENT AGENT PROMPT:
{base_agent_prompt}

FAILED CRITERION:
{task["criterion"]}

REASON FOR FAILURE:
{task["reasoning"] or "Not provided"}

{conversation_text}

Please provide:
1. A modified version of the agent prompt that will fix this issue
2. Specific changes you made
3. Your reasoning for why these changes will work
4. Any tools that should be added (optional)
5. Any MCP servers that should be added (optional)"""

    try:
        # Use structured output with Pydantic model
        completion = await client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": fixer_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=PromptModification,
            temperature=temperature
        )

        # Get parsed Pydantic model
        parsed_response = completion.choices[0].message.parsed

        if not parsed_response:
            return None

        # Extract tool names and MCP server names
        tools_added = None
        if parsed_response.tools_to_add:
            tools_added = [tool.name for tool in parsed_response.tools_to_add]

        mcp_servers_added = None
        if parsed_response.mcp_servers_to_add:
            mcp_servers_added = [server.server_name for server in parsed_response.mcp_servers_to_add]

        # Convert to our internal format
        modification: AgentModification = {
            "criterion": task["criterion"],
            "original_prompt": base_agent_prompt,
            "modified_prompt": parsed_response.modified_prompt,
            "changes_made": parsed_response.changes_made,
            "reasoning": parsed_response.reasoning,
            "tools_added": tools_added,
            "mcp_servers_added": mcp_servers_added
        }

        return modification

    except Exception as e:
        print(f"Error fixing issue for criterion '{task['criterion']}': {e}")
        return None


async def simulate_fixes(
    judgment_result: JudgmentResult,
    base_agent_prompt: str,
    model: str = FIXER_MODEL,
    temperature: float = FIXER_TEMPERATURE,
    parallel: bool = True
) -> SimulationResult:
    """Main primitive for simulation layer - fixes all failed criteria.

    This function analyzes a JudgmentResult and spawns fixer agents for each
    failed criterion. Each fixer agent works independently to suggest modifications.

    Args:
        judgment_result: Result from judging layer
        base_agent_prompt: Current agent prompt that needs improvement
        model: Model identifier to use for fixer agents
        temperature: Sampling temperature
        parallel: Whether to run fixers in parallel (default True)

    Returns:
        SimulationResult containing all modifications
    """
    # Check if all conversations passed
    if judgment_result["overall_statistics"]["total_passed"] == judgment_result["overall_statistics"]["total_judged"]:
        return {
            "modifications": [],
            "total_issues": 0,
            "total_fixed": 0,
            "status": "Everything works!"
        }

    # Collect all failed criteria with their conversations
    tasks: List[FixerTask] = []

    for judgment in judgment_result["judgments"]:
        if not judgment["overall_pass"]:
            # Find the conversation index
            conv_idx = judgment["conversation_index"]
            conversation = judgment_result["judgments"][conv_idx]["conversation_index"]  # This will be set by caller

            # For each failed criterion in this conversation
            for score in judgment["criteria_scores"]:
                if not score["met"]:
                    task: FixerTask = {
                        "criterion": score["criterion"],
                        "reasoning": score["reasoning"],
                        "conversation": None,  # Will be set by caller
                        "conversation_index": conv_idx
                    }
                    tasks.append(task)

    total_issues = len(tasks)

    if total_issues == 0:
        return {
            "modifications": [],
            "total_issues": 0,
            "total_fixed": 0,
            "status": "Everything works!"
        }

    # Run fixer agents
    modifications: List[AgentModification] = []

    if parallel:
        # Fix all issues concurrently
        fix_tasks = [
            fix_single_issue(
                task=task,
                base_agent_prompt=base_agent_prompt,
                model=model,
                temperature=temperature
            )
            for task in tasks
        ]

        results = await asyncio.gather(*fix_tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Fixer {i} failed with exception: {result}")
                continue
            if result is not None:
                modifications.append(result)
    else:
        # Fix issues sequentially
        for task in tasks:
            result = await fix_single_issue(
                task=task,
                base_agent_prompt=base_agent_prompt,
                model=model,
                temperature=temperature
            )
            if result is not None:
                modifications.append(result)

    total_fixed = len(modifications)

    return {
        "modifications": modifications,
        "total_issues": total_issues,
        "total_fixed": total_fixed,
        "status": f"Fixed {total_fixed}/{total_issues} issues"
    }


async def simulate_fixes_from_conversations(
    judgment_result: JudgmentResult,
    conversations: List[Conversation],
    base_agent_prompt: str,
    **kwargs
) -> SimulationResult:
    """Convenience function that takes conversations list along with judgment result.

    This is useful because JudgmentResult doesn't contain the actual conversations,
    only the judgments.

    Args:
        judgment_result: Result from judging layer
        conversations: Original conversations that were judged
        base_agent_prompt: Current agent prompt that needs improvement
        **kwargs: Additional arguments passed to simulate_fixes

    Returns:
        SimulationResult containing all modifications
    """
    # Collect all failed criteria with their conversations
    tasks: List[FixerTask] = []

    for judgment in judgment_result["judgments"]:
        if not judgment["overall_pass"]:
            # Find the conversation
            conv_idx = judgment["conversation_index"]
            conversation = conversations[conv_idx]

            # For each failed criterion in this conversation
            for score in judgment["criteria_scores"]:
                if not score["met"]:
                    task: FixerTask = {
                        "criterion": score["criterion"],
                        "reasoning": score["reasoning"],
                        "conversation": conversation,
                        "conversation_index": conv_idx
                    }
                    tasks.append(task)

    total_issues = len(tasks)

    if total_issues == 0:
        return {
            "modifications": [],
            "total_issues": 0,
            "total_fixed": 0,
            "status": "Everything works!"
        }

    # Run fixer agents
    modifications: List[AgentModification] = []
    parallel = kwargs.get("parallel", True)
    model = kwargs.get("model", FIXER_MODEL)
    temperature = kwargs.get("temperature", FIXER_TEMPERATURE)

    if parallel:
        # Fix all issues concurrently
        fix_tasks = [
            fix_single_issue(
                task=task,
                base_agent_prompt=base_agent_prompt,
                model=model,
                temperature=temperature
            )
            for task in tasks
        ]

        results = await asyncio.gather(*fix_tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Fixer {i} failed with exception: {result}")
                continue
            if result is not None:
                modifications.append(result)
    else:
        # Fix issues sequentially
        for task in tasks:
            result = await fix_single_issue(
                task=task,
                base_agent_prompt=base_agent_prompt,
                model=model,
                temperature=temperature
            )
            if result is not None:
                modifications.append(result)

    total_fixed = len(modifications)

    return {
        "modifications": modifications,
        "total_issues": total_issues,
        "total_fixed": total_fixed,
        "status": f"Fixed {total_fixed}/{total_issues} issues"
    }


# Synchronous wrapper for convenience
def simulate_fixes_sync(
    judgment_result: JudgmentResult,
    conversations: List[Conversation],
    base_agent_prompt: str,
    **kwargs
) -> SimulationResult:
    """Synchronous wrapper for simulate_fixes_from_conversations.

    Args:
        judgment_result: Result from judging layer
        conversations: Original conversations that were judged
        base_agent_prompt: Current agent prompt that needs improvement
        **kwargs: Additional arguments passed to simulate_fixes_from_conversations

    Returns:
        SimulationResult containing all modifications
    """
    return asyncio.run(simulate_fixes_from_conversations(
        judgment_result=judgment_result,
        conversations=conversations,
        base_agent_prompt=base_agent_prompt,
        **kwargs
    ))
