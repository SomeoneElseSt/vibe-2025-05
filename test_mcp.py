
import asyncio
import os
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv

load_dotenv("src/.env")

async def test_weather_mcp():
    print("Testing open-meteo-mcp...")
    
    # Should pick up API key from env now
    client = AsyncDedalus()
    runner = DedalusRunner(client)
    
    prompt = "You are a weather assistant. Use the weather tools to find the weather in San Francisco."
    
    try:
        print("Running agent with open-meteo-mcp...")
        result = await runner.run(
            input="What's the weather in San Francisco?",
            model="openai/gpt-4o-mini",
            mcp_servers=["cathy-di/open-meteo-mcp"],
            max_steps=5
        )
        
        print(f"Agent response: {result.final_output}")
        
        # Check if it looks like it got weather data
        if "temperature" in result.final_output.lower() or "degree" in result.final_output.lower():
            print("SUCCESS: Seems to have accessed weather data")
        else:
            print("FAILURE: Doesn't look like it accessed weather data")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_weather_mcp())

