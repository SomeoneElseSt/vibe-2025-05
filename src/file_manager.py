"""File manager for reading and writing Dedalus agent files.

This module handles reading/writing actual Dedalus agent Python files,
including prompt updates, tool definitions, and MCP server configurations.
"""

import sys
from typing import List, Optional, Dict
from pathlib import Path


def generate_tool_code(tool_name: str, tool_description: str) -> str:
    """Generate Python code for a tool function.

    Args:
        tool_name: Name of the tool function
        tool_description: Description of what the tool does

    Returns:
        Python code string for the tool function
    """
    # Generate a basic tool function template
    function_code = f'''def {tool_name}(query: str) -> str:
    """{tool_description}"""
    # TODO: Implement {tool_name} functionality
    return f"Executed {tool_name} with query: {{query}}"
'''
    return function_code


def generate_agent_file(
    file_path: str,
    prompt: str,
    tools: Optional[List[Dict[str, str]]] = None,
    mcp_servers: Optional[List[str]] = None,
    model: str = "openai/gpt-4o-mini",
    max_turns: int = 10
) -> str:
    """Generate a complete Dedalus agent Python file.

    Args:
        file_path: Path where the file will be written
        prompt: System prompt for the agent
        tools: List of tool definitions with 'name' and 'description'
        mcp_servers: List of MCP server names to enable
        model: Model to use for the agent
        max_turns: Maximum conversation turns

    Returns:
        Complete Python file content as string
    """
    tools = tools or []
    mcp_servers = mcp_servers or []

    # Generate tool function definitions
    tool_definitions = []
    tool_names = []
    for tool in tools:
        tool_code = generate_tool_code(tool['name'], tool['description'])
        tool_definitions.append(tool_code)
        tool_names.append(tool['name'])

    tools_section = '\n\n'.join(tool_definitions) if tool_definitions else '# No custom tools defined'
    tools_list = f"[{', '.join(tool_names)}]" if tool_names else "[]"
    mcp_list = str(mcp_servers) if mcp_servers else "[]"

    # Escape the prompt for Python string literal
    prompt_escaped = prompt.replace('"""', r'\"\"\"').replace('\\', r'\\')

    # Generate complete file
    file_content = f'''"""Dedalus Agent - Auto-generated"""

import asyncio
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv

load_dotenv()

# ============================================
# TOOL DEFINITIONS
# ============================================

{tools_section}


# ============================================
# AGENT CONFIGURATION
# ============================================

AGENT_PROMPT = """{prompt_escaped}"""

MODEL = "{model}"
MAX_TURNS = {max_turns}
TOOLS = {tools_list}
MCP_SERVERS = {mcp_list}


# ============================================
# AGENT RUNNER
# ============================================

async def run_agent(user_message: str) -> str:
    """Run the agent with a user message.

    Args:
        user_message: The user's input message

    Returns:
        The agent's response
    """
    client = AsyncDedalus()
    runner = DedalusRunner(client)

    result = await runner.run(
        input=user_message,
        model=MODEL,
        tools=TOOLS,
        mcp_servers=MCP_SERVERS,
        max_steps=MAX_TURNS
    )

    return result.final_output


async def main():
    """Example usage of the agent."""
    response = await run_agent("Hello, how can you help me?")
    print(f"Agent: {{response}}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
'''

    return file_content


def write_agent_file(
    file_path: str,
    prompt: str,
    tools: Optional[List[Dict[str, str]]] = None,
    mcp_servers: Optional[List[str]] = None,
    **kwargs
) -> str:
    """Write a Dedalus agent file to disk.

    Args:
        file_path: Path where to write the agent file
        prompt: System prompt for the agent
        tools: List of tool definitions
        mcp_servers: List of MCP server names
        **kwargs: Additional arguments passed to generate_agent_file

    Returns:
        The file path that was written
    """
    content = generate_agent_file(
        file_path=file_path,
        prompt=prompt,
        tools=tools,
        mcp_servers=mcp_servers,
        **kwargs
    )

    # Ensure directory exists
    path_obj = Path(file_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Write file
    with open(file_path, 'w') as f:
        f.write(content)

    return file_path


def read_agent_file(file_path: str) -> Dict[str, any]:
    """Read a Dedalus agent file and extract configuration.

    Args:
        file_path: Path to the agent file

    Returns:
        Dictionary with extracted configuration:
        - prompt: Agent system prompt
        - tools: List of tool names
        - mcp_servers: List of MCP server names
        - file_content: Full file content
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # Simple parsing - extract the AGENT_PROMPT
    prompt = ""
    if 'AGENT_PROMPT = """' in content:
        start = content.index('AGENT_PROMPT = """') + len('AGENT_PROMPT = """')
        end = content.index('"""', start)
        prompt = content[start:end]

    # Extract MCP_SERVERS list
    mcp_servers = []
    if 'MCP_SERVERS = [' in content:
        start = content.index('MCP_SERVERS = [') + len('MCP_SERVERS = [')
        end = content.index(']', start)
        mcp_str = content[start:end]
        # Parse the list (simple parsing)
        if mcp_str.strip():
            mcp_servers = [s.strip().strip('"').strip("'") for s in mcp_str.split(',')]

    # Extract TOOLS list (tool names)
    tools = []
    if 'TOOLS = [' in content:
        start = content.index('TOOLS = [') + len('TOOLS = [')
        end = content.index(']', start)
        tools_str = content[start:end]
        if tools_str.strip():
            tools = [s.strip() for s in tools_str.split(',')]

    return {
        "prompt": prompt,
        "tools": tools,
        "mcp_servers": mcp_servers,
        "file_content": content
    }


def apply_modification_to_file(
    file_path: str,
    modification: Dict[str, any]
) -> str:
    """Apply an AgentModification to a Dedalus agent file.

    Args:
        file_path: Path to the agent file to modify
        modification: AgentModification dictionary with:
            - modified_prompt: New prompt text
            - tools_added: List of tool names (optional)
            - mcp_servers_added: List of MCP server names (optional)

    Returns:
        Path to the modified file
    """
    # Read current file
    current = read_agent_file(file_path)

    # Get new values
    new_prompt = modification.get("modified_prompt", current["prompt"])

    # Collect tools to add
    new_tools = []
    if modification.get("tools_added"):
        for tool_name in modification["tools_added"]:
            new_tools.append({
                "name": tool_name,
                "description": f"Tool for {tool_name} functionality"
            })

    # Collect MCP servers to add
    new_mcps = list(current.get("mcp_servers", []))
    if modification.get("mcp_servers_added"):
        for mcp in modification["mcp_servers_added"]:
            if mcp not in new_mcps:
                new_mcps.append(mcp)

    # Write updated file
    return write_agent_file(
        file_path=file_path,
        prompt=new_prompt,
        tools=new_tools,
        mcp_servers=new_mcps
    )
