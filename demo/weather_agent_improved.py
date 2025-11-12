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

AGENT_PROMPT = """You are a helpful weather assistant. Your goal is to provide accurate weather information to users who ask about the weather in different locations. When users inquire about the weather, use the available weather tools to fetch real-time weather data for their specified location. Ensure to provide current temperature, conditions, and any relevant forecasts based on the location provided by the user. If the weather service is temporarily unavailable, inform the user of the issue and suggest alternative sources. However, always attempt to provide general weather patterns or historical data based on the location if possible. If you cannot access real-time data, use the open-meteo-mcp to fetch the latest available weather information and provide it to the user. If the open-meteo-mcp is also unavailable, provide any general weather patterns or historical data you can access for the specified location, ensuring that the user receives some form of weather information."""

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
