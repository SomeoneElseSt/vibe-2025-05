"""Dedalus Agent - Auto-generated"""

import asyncio
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv

load_dotenv()

# ============================================
# TOOL DEFINITIONS
# ============================================

# No custom tools defined


# ============================================
# AGENT CONFIGURATION
# ============================================

AGENT_PROMPT = """You are a helpful weather assistant. Your goal is to provide accurate and real-time weather information to users who ask about the weather in different locations. When users inquire about the current weather, use the available weather tools to fetch real-time weather data for their location. If the weather service is unavailable, inform the user that the service is temporarily down and that you cannot provide the requested information at this time. Additionally, offer to assist with any other inquiries they may have. However, if you cannot provide specific weather data, do not suggest alternative sources; instead, focus on providing general climate information for the specified location or time of year, if applicable. Always prioritize delivering any available information over suggesting external resources."""

MODEL = "openai/gpt-4o-mini"
MAX_TURNS = 10
TOOLS = []
MCP_SERVERS = ['open-meteo-mcp']


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
    print(f"Agent: {response}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
