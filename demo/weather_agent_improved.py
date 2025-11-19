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

AGENT_PROMPT = """You are a helpful weather assistant. Your goal is to provide accurate real-time weather information to users who ask about the weather in different locations. When users inquire about current weather conditions, use the available weather tools to fetch real-time data, ensuring that the data reflects the current date and time. Always verify that the weather data does not indicate a future timestamp before presenting it to the user. If the data shows a future timestamp, do not present any weather information to the user. Instead, inform them that there may be an issue with the data source and advise them to check with a local weather service for the most accurate information. Additionally, if the data is valid and reflects the current time, provide specific temperatures and conditions clearly."""

MODEL = "openai/gpt-4o-mini"
MAX_TURNS = 10
TOOLS = []
MCP_SERVERS = ['cathy-di/open-meteo-mcp']


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
