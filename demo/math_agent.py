"""Dedalus Math Agent - Demo (Initial State - No Tools)

This agent is designed to help users with mathematical calculations,
but initially has no tools to perform complex calculations.
This demonstrates the "before" state in our improvement demo.
"""

import asyncio
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv

load_dotenv()

# ============================================
# AGENT CONFIGURATION
# ============================================

AGENT_PROMPT = """You are a helpful math assistant. You help users with mathematical calculations and number theory problems."""

MODEL = "openai/gpt-4o-mini"
MAX_TURNS = 10
TOOLS = []
MCP_SERVERS = []


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
    response = await run_agent("What are the prime factors of 84?")
    print(f"Agent: {response}")


if __name__ == "__main__":
    asyncio.run(main())