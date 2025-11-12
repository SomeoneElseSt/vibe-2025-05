"""Orchestration module for coordinating the improvement loop.

This module provides primitives for orchestrating the full improvement cycle:
conversations → judge → simulate → merge → apply → repeat until all criteria pass.
"""

import asyncio
import sys
from typing import List, Optional
from dotenv import load_dotenv

from .types import (
    IterationResult,
    OrchestrationResult,
    AgentModification
)
from .conversations import create_multiple_conversations
from .judge import judge_conversation_result
from .simulator import simulate_fixes_from_conversations
from .merger import merge_simulation_result
from .file_manager import (
    read_agent_file,
    write_agent_file,
    apply_modification_to_file
)
from .config import (
    CONVERSATION_MODEL,
    DEFAULT_MAX_TURNS,
    DEFAULT_TEMPERATURE,
    DEFAULT_MODEL
)

# Load environment variables
load_dotenv()

# Constants
DEFAULT_MAX_ITERATIONS = 5


async def run_single_iteration(
    current_prompt: str,
    conversational_prompts: List[str],
    criteria: List[str],
    initial_message: str,
    judge_prompt: str,
    iteration_number: int,
    conversation_model: str = CONVERSATION_MODEL,
    max_turns: int = DEFAULT_MAX_TURNS,
    temperature: float = DEFAULT_TEMPERATURE,
    judge_model: str = DEFAULT_MODEL,
    include_reasoning: bool = True,
    mcp_servers: Optional[List[str]] = None
):
    """Run a single iteration of the improvement loop.
    
    Args:
        current_prompt: Current base agent prompt to test
        conversational_prompts: List of conversational agent prompts
        criteria: List of criteria to judge against
        initial_message: Starting message for conversations
        judge_prompt: System prompt for judge agent
        iteration_number: Current iteration number
        conversation_model: Model for conversations
        max_turns: Max turns per conversation
        temperature: Sampling temperature
        judge_model: Model for judging
        include_reasoning: Include reasoning in judgments
        
    Returns:
        Tuple of (IterationResult, List[Conversation]) or (None, None) if failed
    """
    # Step 1: Create conversations with current prompt and MCP servers
    conversation_result = await create_multiple_conversations(
        base_agent_prompt=current_prompt,
        conversational_agent_prompts=conversational_prompts,
        initial_message=initial_message,
        model=conversation_model,
        max_turns=max_turns,
        temperature=temperature,
        mcp_servers=mcp_servers
    )
    
    if not conversation_result or not conversation_result["conversations"]:
        return None, None
    
    conversations = conversation_result["conversations"]
    
    # Step 2: Judge the conversations
    judgment_result = await judge_conversation_result(
        conversation_result=conversation_result,
        criteria=criteria,
        judge_prompt=judge_prompt,
        model=judge_model,
        include_reasoning=include_reasoning
    )
    
    if not judgment_result:
        return None, None
    
    # Extract statistics
    stats = judgment_result["overall_statistics"]
    total_conversations = stats["total_conversations"]
    total_passed = stats["total_passed"]
    all_criteria_passed = (total_passed == total_conversations)
    
    # Create iteration result
    iteration_result: IterationResult = {
        "iteration": iteration_number,
        "prompt": current_prompt,
        "all_criteria_passed": all_criteria_passed,
        "total_conversations": total_conversations,
        "total_passed": total_passed,
        "modification_applied": None,
        "judgment_result": judgment_result
    }
    
    return iteration_result, conversations


async def orchestrate_improvement(
    initial_prompt: str,
    conversational_prompts: List[str],
    criteria: List[str],
    initial_message: str,
    judge_prompt: str,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    conversation_model: str = CONVERSATION_MODEL,
    max_turns: int = DEFAULT_MAX_TURNS,
    temperature: float = DEFAULT_TEMPERATURE,
    judge_model: str = DEFAULT_MODEL,
    fixer_model: str = DEFAULT_MODEL,
    include_reasoning: bool = True
) -> OrchestrationResult:
    """Main orchestration loop - coordinates full improvement cycle.
    
    This is the main primitive for the orchestration layer. It runs the
    improvement loop until all criteria pass or max iterations is reached.
    
    Loop Logic:
    1. Run conversations with current prompt
    2. Judge the conversations
    3. If all criteria pass → SUCCESS, return
    4. If any criteria fail → simulate fixes (N fixer agents in parallel)
    5. Merge all modifications into single prompt (automatic or LLM-assisted)
    6. Apply merged modification
    7. Repeat from step 1 with modified prompt
    
    Args:
        initial_prompt: Starting base agent prompt
        conversational_prompts: List of conversational agent prompts
        criteria: List of criteria to judge against
        initial_message: Starting message for conversations
        judge_prompt: System prompt for judge agent
        max_iterations: Maximum number of improvement iterations
        conversation_model: Model for conversations
        max_turns: Max turns per conversation
        temperature: Sampling temperature
        judge_model: Model for judging
        fixer_model: Model for fixer agents
        include_reasoning: Include reasoning in judgments
        
    Returns:
        OrchestrationResult with final prompt and iteration history
    """
    iterations: List[IterationResult] = []
    current_prompt = initial_prompt
    
    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{max_iterations} ---", file=sys.stderr)
        
        # Run single iteration
        iteration_result, conversations = await run_single_iteration(
            current_prompt=current_prompt,
            conversational_prompts=conversational_prompts,
            criteria=criteria,
            initial_message=initial_message,
            judge_prompt=judge_prompt,
            iteration_number=iteration,
            conversation_model=conversation_model,
            max_turns=max_turns,
            temperature=temperature,
            judge_model=judge_model,
            include_reasoning=include_reasoning
        )
        
        if not iteration_result or not conversations:
            print(f"Iteration {iteration} failed", file=sys.stderr)
            continue
        
        # Check if all criteria passed
        all_passed = iteration_result["all_criteria_passed"]
        total_passed = iteration_result["total_passed"]
        total_conversations = iteration_result["total_conversations"]
        
        print(f"Result: {total_passed}/{total_conversations} conversations passed all criteria", file=sys.stderr)
        
        # Store iteration result (without modification yet)
        iterations.append(iteration_result)
        
        # If all criteria passed, we're done
        if all_passed:
            print("SUCCESS: All criteria passed!", file=sys.stderr)
            return {
                "success": True,
                "final_prompt": current_prompt,
                "all_criteria_passed": True,
                "iterations": iterations,
                "total_iterations": iteration,
                "status": f"Success after {iteration} iteration(s)"
            }
        
        # Not all passed - simulate fixes
        print("Simulating fixes for failed criteria...", file=sys.stderr)
        simulation_result = await simulate_fixes_from_conversations(
            judgment_result=iteration_result["judgment_result"],
            conversations=conversations,
            base_agent_prompt=current_prompt,
            model=fixer_model
        )

        if not simulation_result or not simulation_result["modifications"]:
            print("No modifications generated - stopping", file=sys.stderr)
            break

        num_mods = len(simulation_result["modifications"])
        print(f"Generated {num_mods} modification(s)", file=sys.stderr)

        # Merge all modifications into a single prompt
        print("Merging modifications...", file=sys.stderr)
        merge_result = await merge_simulation_result(
            simulation_result=simulation_result,
            original_prompt=current_prompt,
            model=fixer_model
        )

        print(f"Merge method: {merge_result['merge_method']}", file=sys.stderr)
        if merge_result['had_conflicts']:
            print(f"Resolved {merge_result['conflicts_resolved']} conflict(s) using LLM", file=sys.stderr)

        # For tracking purposes, store the first modification (or create a merged one)
        if num_mods == 1:
            modification = simulation_result["modifications"][0]
        else:
            # Create a synthetic modification representing the merge
            modification: AgentModification = {
                "criterion": f"Merged {num_mods} modifications",
                "original_prompt": current_prompt,
                "modified_prompt": merge_result["merged_prompt"],
                "changes_made": [f"Merged {num_mods} modifications using {merge_result['merge_method']} strategy"],
                "reasoning": f"Combined fixes for multiple failed criteria",
                "tools_added": None,
                "mcp_servers_added": None
            }

        print(f"Applying merged modification", file=sys.stderr)

        # Update iteration result with applied modification
        iteration_result["modification_applied"] = modification

        # Update current prompt for next iteration using merged result
        current_prompt = merge_result["merged_prompt"]
    
    # Max iterations reached without success
    final_iteration = iterations[-1] if iterations else None
    all_passed = final_iteration["all_criteria_passed"] if final_iteration else False
    
    return {
        "success": False,
        "final_prompt": current_prompt,
        "all_criteria_passed": all_passed,
        "iterations": iterations,
        "total_iterations": len(iterations),
        "status": f"Max iterations ({max_iterations}) reached"
    }


# Synchronous wrapper for convenience
def orchestrate_improvement_sync(
    initial_prompt: str,
    conversational_prompts: List[str],
    criteria: List[str],
    initial_message: str,
    judge_prompt: str,
    **kwargs
) -> OrchestrationResult:
    """Synchronous wrapper for orchestrate_improvement.

    Args:
        initial_prompt: Starting base agent prompt
        conversational_prompts: List of conversational agent prompts
        criteria: List of criteria to judge against
        initial_message: Starting message for conversations
        judge_prompt: System prompt for judge agent
        **kwargs: Additional arguments passed to orchestrate_improvement

    Returns:
        OrchestrationResult with final prompt and iteration history
    """
    return asyncio.run(orchestrate_improvement(
        initial_prompt=initial_prompt,
        conversational_prompts=conversational_prompts,
        criteria=criteria,
        initial_message=initial_message,
        judge_prompt=judge_prompt,
        **kwargs
    ))


async def orchestrate_improvement_with_file(
    initial_agent_file: str,
    conversational_prompts: List[str],
    criteria: List[str],
    initial_message: str,
    judge_prompt: str,
    output_file: Optional[str] = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    **kwargs
) -> OrchestrationResult:
    """File-based orchestration - modifies actual Dedalus agent files.

    This version reads/writes actual agent files, applying modifications
    with tools and MCP servers to the file on disk.

    Args:
        initial_agent_file: Path to initial Dedalus agent file
        conversational_prompts: List of conversational agent prompts
        criteria: List of criteria to judge against
        initial_message: Starting message for conversations
        judge_prompt: System prompt for judge agent
        output_file: Path for output file (defaults to initial_agent_file)
        max_iterations: Maximum number of improvement iterations
        **kwargs: Additional arguments (model, temperature, etc.)

    Returns:
        OrchestrationResult with final file path
    """
    output_file = output_file or initial_agent_file

    # Read initial agent file
    print(f"Reading initial agent file: {initial_agent_file}", file=sys.stderr, flush=True)
    agent_config = read_agent_file(initial_agent_file)
    current_prompt = agent_config["prompt"]
    
    print(f"Initial prompt: {current_prompt[:100]}...", file=sys.stderr, flush=True)
    print(f"Initial MCPs: {agent_config.get('mcp_servers', [])}", file=sys.stderr, flush=True)
    print(f"Initial tools: {agent_config.get('tools', [])}", file=sys.stderr, flush=True)

    # Track accumulated MCP servers and tools across iterations
    accumulated_mcps = list(agent_config.get('mcp_servers', []))
    accumulated_tools = []
    current_file_path = initial_agent_file

    # Run orchestration loop iteration by iteration with file updates
    iterations: List[IterationResult] = []
    
    for iteration_num in range(1, max_iterations + 1):
        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print(f"FILE-BASED ITERATION {iteration_num}/{max_iterations}", file=sys.stderr, flush=True)
        print(f"{'='*60}", file=sys.stderr, flush=True)
        print(f"Using file: {current_file_path}", file=sys.stderr, flush=True)
        print(f"Current MCPs: {accumulated_mcps}", file=sys.stderr, flush=True)
        print(f"Current prompt (first 100 chars): {current_prompt[:100]}...", file=sys.stderr, flush=True)
        
        # Run ONE iteration with accumulated MCP servers
        iteration_result, conversations = await run_single_iteration(
            current_prompt=current_prompt,
            conversational_prompts=conversational_prompts,
            criteria=criteria,
            initial_message=initial_message,
            judge_prompt=judge_prompt,
            iteration_number=iteration_num,
            mcp_servers=accumulated_mcps if accumulated_mcps else None,
            **kwargs
        )
        
        if not iteration_result or not conversations:
            print(f"Iteration {iteration_num} failed", file=sys.stderr, flush=True)
            break
        
        all_passed = iteration_result["all_criteria_passed"]
        
        # Store iteration
        iterations.append(iteration_result)
        
        print(f"Iteration {iteration_num} result: {iteration_result['total_passed']}/{iteration_result['total_conversations']} passed", file=sys.stderr, flush=True)
        
        if all_passed:
            print("SUCCESS: All criteria passed!", file=sys.stderr, flush=True)
            break
        
        # Get fixes
        print("Simulating fixes...", file=sys.stderr, flush=True)
        simulation_result = await simulate_fixes_from_conversations(
            judgment_result=iteration_result["judgment_result"],
            conversations=conversations,
            base_agent_prompt=current_prompt,
            **{k: v for k, v in kwargs.items() if k in ['model', 'temperature']}
        )
        
        if not simulation_result or not simulation_result["modifications"]:
            print("No modifications generated", file=sys.stderr, flush=True)
            break
        
        # Merge modifications
        print(f"Merging {len(simulation_result['modifications'])} modification(s)...", file=sys.stderr, flush=True)
        merge_result = await merge_simulation_result(
            simulation_result=simulation_result,
            original_prompt=current_prompt,
            **{k: v for k, v in kwargs.items() if k in ['model', 'temperature']}
        )
        
        print(f"Merge method: {merge_result['merge_method']}", file=sys.stderr, flush=True)
        
        # Track ONE modification for iteration result
        if simulation_result["modifications"]:
            iteration_result["modification_applied"] = simulation_result["modifications"][0]
        
        # Accumulate MCPs and tools from ALL modifications
        for mod in simulation_result["modifications"]:
            if mod.get("mcp_servers_added"):
                for mcp in mod["mcp_servers_added"]:
                    if mcp not in accumulated_mcps:
                        accumulated_mcps.append(mcp)
                        print(f"  + Adding MCP: {mcp}", file=sys.stderr, flush=True)
            if mod.get("tools_added"):
                for tool in mod["tools_added"]:
                    if tool not in accumulated_tools:
                        accumulated_tools.append(tool)
                        print(f"  + Adding tool: {tool}", file=sys.stderr, flush=True)
        
        # Update prompt
        current_prompt = merge_result["merged_prompt"]
        
        # Write improved file with accumulated MCPs/tools
        print(f"\nWriting improved file to: {output_file}", file=sys.stderr, flush=True)
        print(f"  Prompt (first 100 chars): {current_prompt[:100]}...", file=sys.stderr, flush=True)
        print(f"  MCPs to write: {accumulated_mcps}", file=sys.stderr, flush=True)
        print(f"  Tools to write: {accumulated_tools}", file=sys.stderr, flush=True)
        
        tool_defs = [{"name": t, "description": f"Tool for {t}"} for t in accumulated_tools]
        
        write_agent_file(
            file_path=output_file,
            prompt=current_prompt,
            tools=tool_defs if tool_defs else None,
            mcp_servers=accumulated_mcps if accumulated_mcps else None
        )
        
        # CRITICAL: Update current_file_path to point to improved file for next iteration
        current_file_path = output_file
        
        # Verify what was written
        written_config = read_agent_file(output_file)
        print(f"\nVERIFYING WRITTEN FILE:", file=sys.stderr, flush=True)
        print(f"  Read prompt (first 100 chars): {written_config['prompt'][:100]}...", file=sys.stderr, flush=True)
        print(f"  Read MCPs: {written_config.get('mcp_servers', [])}", file=sys.stderr, flush=True)
        print(f"  Read tools: {written_config.get('tools', [])}", file=sys.stderr, flush=True)
    
    # Build result
    all_passed = iterations[-1]["all_criteria_passed"] if iterations else False
    
    result: OrchestrationResult = {
        "success": all_passed,
        "final_prompt": current_prompt,
        "all_criteria_passed": all_passed,
        "iterations": iterations,
        "total_iterations": len(iterations),
        "status": f"Success after {len(iterations)} iteration(s)" if all_passed else f"Max iterations ({max_iterations}) reached",
        "final_agent_file": output_file
    }
    
    return result

    # If modifications were made, write the final file
    if result["iterations"]:
        final_iteration = result["iterations"][-1]

        if final_iteration.get("modification_applied"):
            print(f"\nWriting final agent file to: {output_file}", file=sys.stderr)
            modification = final_iteration["modification_applied"]

            # Collect all tools and MCPs from all iterations
            all_tools = []
            all_mcps = []

            for iteration in result["iterations"]:
                if iteration.get("modification_applied"):
                    mod = iteration["modification_applied"]
                    if mod.get("tools_added"):
                        for tool in mod["tools_added"]:
                            if tool not in all_tools:
                                all_tools.append(tool)
                    if mod.get("mcp_servers_added"):
                        for mcp in mod["mcp_servers_added"]:
                            if mcp not in all_mcps:
                                all_mcps.append(mcp)

            # Create tool definitions
            tool_defs = [{"name": t, "description": f"Tool for {t}"} for t in all_tools]

            # Write final file with accumulated changes
            final_file = write_agent_file(
                file_path=output_file,
                prompt=result["final_prompt"],
                tools=tool_defs if tool_defs else None,
                mcp_servers=all_mcps if all_mcps else None
            )

            print(f"✅ Final agent file written successfully", file=sys.stderr)
            print(f"   Tools added: {all_tools}", file=sys.stderr)
            print(f"   MCPs added: {all_mcps}", file=sys.stderr)

            result["final_agent_file"] = final_file
        else:
            # No modifications, just write the final prompt
            agent_config = read_agent_file(initial_agent_file)
            final_file = write_agent_file(
                file_path=output_file,
                prompt=result["final_prompt"],
                tools=None,
                mcp_servers=agent_config.get("mcp_servers")
            )
            result["final_agent_file"] = final_file
    else:
        result["final_agent_file"] = None

    return result


def orchestrate_improvement_with_file_sync(
    initial_agent_file: str,
    conversational_prompts: List[str],
    criteria: List[str],
    initial_message: str,
    judge_prompt: str,
    **kwargs
) -> OrchestrationResult:
    """Synchronous wrapper for file-based orchestration.

    Args:
        initial_agent_file: Path to initial Dedalus agent file
        conversational_prompts: List of conversational agent prompts
        criteria: List of criteria to judge against
        initial_message: Starting message for conversations
        judge_prompt: System prompt for judge agent
        **kwargs: Additional arguments

    Returns:
        OrchestrationResult with final file path
    """
    return asyncio.run(orchestrate_improvement_with_file(
        initial_agent_file=initial_agent_file,
        conversational_prompts=conversational_prompts,
        criteria=criteria,
        initial_message=initial_message,
        judge_prompt=judge_prompt,
        **kwargs
    ))