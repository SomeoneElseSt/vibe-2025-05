"""Judging module for evaluating agent conversations.

This module provides primitives for evaluating conversations against criteria.
"""

import asyncio
import sys
from typing import List, Optional, Dict
from dotenv import load_dotenv
from pydantic import BaseModel
from dedalus_labs import AsyncDedalus

from .types import (
    Conversation,
    ConversationResult,
    CriterionScore,
    ConversationJudgment,
    JudgmentResult
)
from .config import DEFAULT_MODEL, DEFAULT_TEMPERATURE

# Load environment variables
load_dotenv()


# Pydantic models for structured output
class CriterionEvaluation(BaseModel):
    """Evaluation of a single criterion."""
    criterion: str
    met: bool
    reasoning: Optional[str] = None


class EvaluationResponse(BaseModel):
    """Structured response from judge."""
    evaluations: List[CriterionEvaluation]


def format_conversation_for_judge(conversation: Conversation) -> str:
    """Format a conversation into a readable string for the judge.

    Args:
        conversation: Conversation to format

    Returns:
        Formatted string representation
    """
    lines = []
    lines.append("=== CONVERSATION ===")

    for i, message in enumerate(conversation["messages"]):
        role = message["role"]
        content = message["content"]
        lines.append(f"\nTurn {i+1} - {role}:")
        lines.append(content)

    lines.append("\n=== END CONVERSATION ===")
    return "\n".join(lines)


async def judge_single_conversation(
    conversation: Conversation,
    criteria: List[str],
    judge_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    include_reasoning: bool = False
) -> Optional[ConversationJudgment]:
    """Judge a single conversation against multiple criteria using structured outputs.

    Args:
        conversation: The conversation to judge
        criteria: List of criteria to evaluate (e.g., ["Offered payment agreement", "Showed empathy"])
        judge_prompt: System prompt for the judge agent
        model: Model identifier to use
        temperature: Sampling temperature
        include_reasoning: Whether to ask for reasoning for each criterion

    Returns:
        ConversationJudgment or None if failed
    """
    client = AsyncDedalus()

    # Format conversation
    conversation_text = format_conversation_for_judge(conversation)

    # Build criteria list
    criteria_list = "\n".join([f"{i+1}. {criterion}" for i, criterion in enumerate(criteria)])

    # Create evaluation prompt
    reasoning_instruction = "Provide brief reasoning for each evaluation." if include_reasoning else ""

    evaluation_prompt = f"""Evaluate this conversation against the criteria below:

{conversation_text}

CRITERIA TO EVALUATE:
{criteria_list}

For each criterion, determine if it was met (true/false).
{reasoning_instruction}"""

    try:
        print(f"\nJUDGING CONVERSATION...", file=sys.stderr, flush=True)
        print(f"Criteria: {criteria}", file=sys.stderr, flush=True)
        
        # Use structured output with Pydantic model
        completion = await client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": judge_prompt},
                {"role": "user", "content": evaluation_prompt}
            ],
            response_format=EvaluationResponse,
            temperature=temperature
        )

        # Get parsed Pydantic model
        parsed_response = completion.choices[0].message.parsed

        if not parsed_response:
            return None
        
        print(f"\nJUDGMENT RESULTS:", file=sys.stderr, flush=True)
        for eval_item in parsed_response.evaluations:
            status = "✓ PASS" if eval_item.met else "✗ FAIL"
            print(f"  {status}: {eval_item.criterion}", file=sys.stderr, flush=True)
            if eval_item.reasoning:
                print(f"    Reasoning: {eval_item.reasoning}", file=sys.stderr, flush=True)

        # Convert to our internal format
        criteria_scores: List[CriterionScore] = []
        for eval_item in parsed_response.evaluations:
            score: CriterionScore = {
                "criterion": eval_item.criterion,
                "met": eval_item.met,
                "reasoning": eval_item.reasoning if include_reasoning else None
            }
            criteria_scores.append(score)

        # Calculate overall pass (all criteria must be met)
        overall_pass = all(score["met"] for score in criteria_scores)

        judgment: ConversationJudgment = {
            "conversation_index": -1,  # Will be set by caller
            "criteria_scores": criteria_scores,
            "overall_pass": overall_pass
        }

        return judgment

    except Exception as e:
        print(f"Error judging conversation: {e}", file=sys.stderr)
        return None


async def judge_multiple_conversations(
    conversations: List[Conversation],
    criteria: List[str],
    judge_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    include_reasoning: bool = False,
    parallel: bool = True
) -> JudgmentResult:
    """Judge multiple conversations against criteria.

    This is the main primitive for evaluating conversations.

    Args:
        conversations: List of conversations to judge
        criteria: List of criteria to evaluate
        judge_prompt: System prompt for the judge agent
        model: Model identifier to use
        temperature: Sampling temperature
        include_reasoning: Whether to include reasoning for each criterion
        parallel: Whether to judge conversations in parallel (default True)

    Returns:
        JudgmentResult containing all judgments and statistics
    """
    judgments: List[ConversationJudgment] = []

    if parallel:
        # Judge all conversations concurrently
        tasks = [
            judge_single_conversation(
                conversation=conv,
                criteria=criteria,
                judge_prompt=judge_prompt,
                model=model,
                temperature=temperature,
                include_reasoning=include_reasoning
            )
            for conv in conversations
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Judgment {i} failed with exception: {result}", file=sys.stderr)
                continue
            if result is not None:
                result["conversation_index"] = i
                judgments.append(result)
    else:
        # Judge conversations sequentially
        for i, conv in enumerate(conversations):
            result = await judge_single_conversation(
                conversation=conv,
                criteria=criteria,
                judge_prompt=judge_prompt,
                model=model,
                temperature=temperature,
                include_reasoning=include_reasoning
            )
            if result is not None:
                result["conversation_index"] = i
                judgments.append(result)

    # Calculate statistics
    total_judged = len(judgments)
    total_passed = sum(1 for j in judgments if j["overall_pass"])

    # Per-criterion statistics
    criterion_stats = {}
    for criterion in criteria:
        met_count = 0
        for judgment in judgments:
            for score in judgment["criteria_scores"]:
                if score["criterion"] == criterion and score["met"]:
                    met_count += 1

        criterion_stats[criterion] = {
            "met_count": met_count,
            "total": total_judged,
            "percentage": (met_count / total_judged * 100) if total_judged > 0 else 0
        }

    return {
        "judgments": judgments,
        "overall_statistics": {
            "total_conversations": len(conversations),
            "total_judged": total_judged,
            "total_passed": total_passed,
            "pass_rate": (total_passed / total_judged * 100) if total_judged > 0 else 0,
            "criterion_statistics": criterion_stats
        },
        "judge_config": {
            "criteria": criteria,
            "judge_prompt": judge_prompt,
            "model": model,
            "temperature": temperature,
            "include_reasoning": include_reasoning
        }
    }


async def judge_conversation_result(
    conversation_result: ConversationResult,
    criteria: List[str],
    judge_prompt: str,
    **kwargs
) -> JudgmentResult:
    """Convenience function to judge a ConversationResult directly.

    Args:
        conversation_result: Result from create_multiple_conversations
        criteria: List of criteria to evaluate
        judge_prompt: System prompt for the judge agent
        **kwargs: Additional arguments passed to judge_multiple_conversations

    Returns:
        JudgmentResult containing all judgments
    """
    return await judge_multiple_conversations(
        conversations=conversation_result["conversations"],
        criteria=criteria,
        judge_prompt=judge_prompt,
        **kwargs
    )


# Synchronous wrapper for convenience
def judge_conversations_sync(
    conversations: List[Conversation],
    criteria: List[str],
    judge_prompt: str,
    **kwargs
) -> JudgmentResult:
    """Synchronous wrapper for judge_multiple_conversations.

    Args:
        conversations: List of conversations to judge
        criteria: List of criteria to evaluate
        judge_prompt: System prompt for the judge agent
        **kwargs: Additional arguments passed to judge_multiple_conversations

    Returns:
        JudgmentResult containing all judgments
    """
    return asyncio.run(judge_multiple_conversations(
        conversations=conversations,
        criteria=criteria,
        judge_prompt=judge_prompt,
        **kwargs
    ))
