"""FastAPI backend for agent orchestration demo.

Simple API with single endpoint to run orchestration.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import orchestrate_improvement_with_file

app = FastAPI(title="Agent Orchestration API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class OrchestrationRequest(BaseModel):
    """Request model for orchestration endpoint."""
    criteria: List[str]


@app.post("/api/orchestrate")
async def orchestrate(request: OrchestrationRequest):
    """Run orchestration with weather demo scenario.
    
    Args:
        request: Contains SOP criteria to evaluate
        
    Returns:
        Full orchestration result with iterations
    """
    if not request.criteria:
        raise HTTPException(status_code=400, detail="Criteria list cannot be empty")
    
    # Fixed demo scenario
    initial_agent_file = "demo/weather_agent.py"
    output_file = "demo/weather_agent_improved.py"
    
    conversational_prompts = [
        "You are a user asking about the weather. Be conversational and ask for weather in different cities.",
        "You are a user who wants to know the weather forecast. Ask about multiple locations."
    ]
    
    initial_message = "What's the weather like today?"
    judge_prompt = "You are an expert evaluator of weather assistant conversations."
    
    try:
        # Run async orchestration
        result = await orchestrate_improvement_with_file(
            initial_agent_file=initial_agent_file,
            conversational_prompts=conversational_prompts,
            criteria=request.criteria,
            initial_message=initial_message,
            judge_prompt=judge_prompt,
            output_file=output_file,
            max_iterations=3,
            max_turns=4
        )
        
        return result
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR: {error_details}")  # This will show in FastAPI logs
        raise HTTPException(status_code=500, detail=f"Orchestration failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}