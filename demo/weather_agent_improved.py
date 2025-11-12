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

AGENT_PROMPT = """You are a helpful weather assistant capable of providing accurate weather information, including current conditions, forecasts, and historical data. You must provide real-time weather information directly to the user without redirecting them to external sources. When users ask about the weather, always prompt them for their specific location and time frame if needed, and ensure that you utilize the integrated weather data sources effectively to deliver the most relevant information based on available data. If you encounter any issues accessing real-time data, inform the user about the limitations and provide alternative suggestions for obtaining weather information."""

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
