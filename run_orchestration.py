#!/usr/bin/env python3
"""Direct orchestration runner for demo.

Runs orchestration and outputs JSON results.
No FastAPI, no server - just direct execution.
"""

import sys
import json
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

# Load env from src/.env
load_dotenv("src/.env")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import orchestrate_improvement_with_file


async def run_orchestration(criteria: list[str]) -> dict:
    """Run orchestration with weather demo scenario.
    
    Args:
        criteria: List of SOP criteria to evaluate
        
    Returns:
        Orchestration result as dict
    """
    # Print progress to stderr
    print("Starting orchestration...", file=sys.stderr, flush=True)
    
    # Fixed demo scenario
    initial_agent_file = "demo/weather_agent.py"
    output_file = "demo/weather_agent_improved.py"
    
    conversational_prompts = [
        "You are a user asking about current, real-time weather. When asked for your location, specify San Francisco."
    ]
    
    initial_message = "What's the current weather in San Francisco right now?"
    judge_prompt = "You are an expert evaluator of weather assistant conversations. Focus on whether the agent provides REAL-TIME, CURRENT weather data (temperatures, conditions), not general patterns or historical information."
    
    print(f"Evaluating agent against {len(criteria)} criteria", file=sys.stderr, flush=True)
    print("Running improvement loop (max 3 iterations)...", file=sys.stderr, flush=True)
    
    # Redirect stdout to stderr temporarily to suppress orchestrator prints
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    
    try:
        # Run orchestration - will loop until all criteria pass or max iterations
        result = await orchestrate_improvement_with_file(
            initial_agent_file=initial_agent_file,
            conversational_prompts=conversational_prompts,
            criteria=criteria,
            initial_message=initial_message,
            judge_prompt=judge_prompt,
            output_file=output_file,
            max_iterations=3,
            max_turns=6
        )
    finally:
        # Restore stdout
        sys.stdout = old_stdout
    
    print("Orchestration complete!", file=sys.stderr, flush=True)
    
    return result


def main():
    """Main entry point."""
    try:
        # Read criteria from stdin (JSON string)
        input_data = sys.stdin.read()
        data = json.loads(input_data)
        criteria = data.get("criteria", [])
        
        if not criteria:
            print(json.dumps({"error": "No criteria provided"}))
            sys.exit(1)
        
        # Run orchestration
        result = asyncio.run(run_orchestration(criteria))
        
        # Output JSON result
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()