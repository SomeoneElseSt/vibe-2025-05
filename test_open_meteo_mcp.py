#!/usr/bin/env python
"""
Test file to verify Dedalus agent is calling the open-meteo MCP server.
Tests weather querying functionality.

Setup:
1. Set DEDALUS_API_KEY environment variable, or
2. Set OPENAI_API_KEY for BYOK (Bring Your Own Key)
"""

import asyncio
import os
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv

load_dotenv()


async def main():
    """Test open-meteo MCP with Dedalus agent."""
    # Try to get API key from environment
    api_key = os.getenv("DEDALUS_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("ERROR: No API key found!")
        print("Please set one of the following:")
        print("  - DEDALUS_API_KEY (recommended)")
        print("  - OPENAI_API_KEY (for BYOK mode)")
        return

    print(f"Using API key: {api_key[:20]}...")
    client = AsyncDedalus(api_key=api_key)
    runner = DedalusRunner(client)

    # Test 1: Simple weather query using open-meteo MCP
    print("=" * 60)
    print("Test 1: Simple Weather Query")
    print("=" * 60)
    result = await runner.run(
        input="What's the current weather in San Francisco, California?",
        model="openai/gpt-4o-mini",
        mcp_servers=["cathy-di/open-meteo-mcp"],
        max_steps=10
    )
    print(f"Result: {result.final_output}")
    print(f"Tools called: {result.tools_called}")
    print()

    # Test 2: Multiple location weather query
    print("=" * 60)
    print("Test 2: Multiple Location Query")
    print("=" * 60)
    result = await runner.run(
        input="Compare the weather in London, Paris, and Tokyo. Which is warmest?",
        model="openai/gpt-4o-mini",
        mcp_servers=["cathy-di/open-meteo-mcp"],
        max_steps=15
    )
    print(f"Result: {result.final_output}")
    print(f"Tools called: {result.tools_called}")
    print()

    # Test 3: Weather with streaming
    print("=" * 60)
    print("Test 3: Streaming Response")
    print("=" * 60)
    result = runner.run(
        input="Get the weather forecast for New York and tell me if I should bring an umbrella",
        model="openai/gpt-4o-mini",
        mcp_servers=["cathy-di/open-meteo-mcp"],
        stream=True,
        max_steps=10
    )

    async for token in result:
        print(token, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
