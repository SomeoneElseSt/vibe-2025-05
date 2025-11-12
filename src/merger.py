"""Merge module for combining multiple agent modifications.

This module provides primitives for merging multiple prompt modifications,
handling conflicts automatically or using LLM assistance when needed.
"""

import difflib
from typing import List, Optional, Tuple
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from dedalus_labs import AsyncDedalus

from .types import (
    AgentModification,
    SimulationResult,
    MergeConflict,
    MergeResult
)
from .config import FIXER_MODEL, FIXER_TEMPERATURE

# Load environment variables
load_dotenv()

# Constants
LARGE_CHANGE_THRESHOLD = 500  # Characters


# Pydantic model for LLM-assisted merge
class MergedPrompt(BaseModel):
    """Structured response from merge LLM."""
    merged_prompt: str = Field(description="The intelligently merged prompt combining both modifications")
    reasoning: str = Field(description="Explanation of how the modifications were merged")


def calculate_diff_size(original: str, modified: str) -> int:
    """Calculate the size of changes between two strings.

    Args:
        original: Original text
        modified: Modified text

    Returns:
        Approximate number of characters changed
    """
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        lineterm=''
    )

    # Count characters in diff lines (excluding context)
    change_size = 0
    for line in diff:
        if line.startswith('+') or line.startswith('-'):
            if not line.startswith('+++') and not line.startswith('---'):
                change_size += len(line)

    return change_size


def detect_overlap(original: str, mod1: str, mod2: str) -> bool:
    """Detect if two modifications overlap in the text they change.

    Args:
        original: Original text
        mod1: First modification
        mod2: Second modification

    Returns:
        True if modifications overlap, False otherwise
    """
    # Get the diffs for both modifications
    diff1 = list(difflib.unified_diff(
        original.splitlines(keepends=True),
        mod1.splitlines(keepends=True),
        lineterm=''
    ))

    diff2 = list(difflib.unified_diff(
        original.splitlines(keepends=True),
        mod2.splitlines(keepends=True),
        lineterm=''
    ))

    # Extract changed line numbers from both diffs
    def get_changed_lines(diff_lines):
        changed = set()
        for line in diff_lines:
            if line.startswith('@@'):
                # Parse hunk header to get line numbers
                parts = line.split()
                if len(parts) >= 2:
                    # Extract line range from format: @@ -1,3 +1,4 @@
                    try:
                        minus_part = parts[1].lstrip('-').split(',')
                        start = int(minus_part[0])
                        length = int(minus_part[1]) if len(minus_part) > 1 else 1
                        for i in range(start, start + length):
                            changed.add(i)
                    except (ValueError, IndexError):
                        pass
        return changed

    changed1 = get_changed_lines(diff1)
    changed2 = get_changed_lines(diff2)

    # Check for overlap
    return bool(changed1 & changed2)


def apply_non_conflicting_merge(original: str, mod1: str, mod2: str) -> Optional[str]:
    """Attempt to automatically merge two non-conflicting modifications.

    Args:
        original: Original text
        mod1: First modification
        mod2: Second modification

    Returns:
        Merged text if successful, None if conflict detected
    """
    # Check if modifications overlap
    if detect_overlap(original, mod1, mod2):
        return None

    # Use difflib to create a 3-way merge
    # First, apply mod1's changes to original
    diff1 = difflib.unified_diff(
        original.splitlines(keepends=True),
        mod1.splitlines(keepends=True),
        lineterm=''
    )

    # Then, apply mod2's changes to original
    diff2 = difflib.unified_diff(
        original.splitlines(keepends=True),
        mod2.splitlines(keepends=True),
        lineterm=''
    )

    # For non-overlapping changes, we can simply apply both modifications
    # by taking the union of their changes

    # Simple approach: try to merge line-by-line
    original_lines = original.splitlines(keepends=True)
    mod1_lines = mod1.splitlines(keepends=True)
    mod2_lines = mod2.splitlines(keepends=True)

    # Use SequenceMatcher to find matching blocks
    matcher1 = difflib.SequenceMatcher(None, original_lines, mod1_lines)
    matcher2 = difflib.SequenceMatcher(None, original_lines, mod2_lines)

    # Get opcodes (operations needed to transform original to modified)
    opcodes1 = matcher1.get_opcodes()
    opcodes2 = matcher2.get_opcodes()

    # Build merged result
    result = []
    i = 0
    j1 = 0
    j2 = 0

    # Simple merge: if lines are added at different positions, include both
    while i < len(original_lines):
        # Check what mod1 does at this position
        mod1_operation = None
        for tag, i1, i2, j1_start, j1_end in opcodes1:
            if i1 <= i < i2:
                mod1_operation = (tag, i1, i2, j1_start, j1_end)
                break

        # Check what mod2 does at this position
        mod2_operation = None
        for tag, i1, i2, j2_start, j2_end in opcodes2:
            if i1 <= i < i2:
                mod2_operation = (tag, i1, i2, j2_start, j2_end)
                break

        # If both modify the same line, we have a conflict
        if (mod1_operation and mod1_operation[0] != 'equal' and
            mod2_operation and mod2_operation[0] != 'equal'):
            return None  # Conflict detected

        # Apply the modification that exists
        if mod1_operation and mod1_operation[0] == 'replace':
            tag, i1, i2, j1_start, j1_end = mod1_operation
            result.extend(mod1_lines[j1_start:j1_end])
            i = i2
        elif mod2_operation and mod2_operation[0] == 'replace':
            tag, i1, i2, j2_start, j2_end = mod2_operation
            result.extend(mod2_lines[j2_start:j2_end])
            i = i2
        else:
            result.append(original_lines[i])
            i += 1

    return ''.join(result)


async def llm_merge_modifications(
    original: str,
    mod1: AgentModification,
    mod2: AgentModification,
    model: str = FIXER_MODEL,
    temperature: float = FIXER_TEMPERATURE
) -> Optional[str]:
    """Use LLM to intelligently merge two conflicting modifications.

    Args:
        original: Original prompt text
        mod1: First modification
        mod2: Second modification
        model: Model to use for merging
        temperature: Sampling temperature

    Returns:
        Merged prompt text or None if failed
    """
    client = AsyncDedalus()

    system_prompt = """You are an expert at merging agent prompt modifications.
You will be given an original prompt and two different modifications to it.
Your task is to intelligently combine both modifications into a single, coherent prompt
that incorporates the improvements from both changes.

Preserve the intent and improvements from both modifications while ensuring the result is:
1. Coherent and well-structured
2. Free of redundancy
3. Clear and actionable
4. Maintains the original agent's core purpose"""

    user_prompt = f"""Merge these two prompt modifications intelligently.

ORIGINAL PROMPT:
{original}

MODIFICATION 1 (for criterion: {mod1['criterion']}):
{mod1['modified_prompt']}

Changes made in Modification 1:
{chr(10).join(f"- {change}" for change in mod1['changes_made'])}

Reasoning: {mod1['reasoning']}

MODIFICATION 2 (for criterion: {mod2['criterion']}):
{mod2['modified_prompt']}

Changes made in Modification 2:
{chr(10).join(f"- {change}" for change in mod2['changes_made'])}

Reasoning: {mod2['reasoning']}

Please create a merged prompt that incorporates the best aspects of both modifications."""

    try:
        completion = await client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=MergedPrompt,
            temperature=temperature
        )

        parsed_response = completion.choices[0].message.parsed

        if not parsed_response:
            return None

        return parsed_response.merged_prompt

    except Exception as e:
        print(f"Error in LLM merge: {e}")
        return None


async def merge_modifications(
    modifications: List[AgentModification],
    original_prompt: str,
    model: str = FIXER_MODEL,
    temperature: float = FIXER_TEMPERATURE
) -> MergeResult:
    """Main primitive for merging multiple agent modifications.

    This function takes multiple modifications from the simulation layer and
    intelligently merges them into a single prompt. It handles:
    1. Single modification (no merge needed)
    2. Multiple non-conflicting modifications (automatic merge)
    3. Conflicting modifications (LLM-assisted merge)

    Args:
        modifications: List of modifications from simulation layer
        original_prompt: Original agent prompt
        model: Model to use for LLM-assisted merges
        temperature: Sampling temperature

    Returns:
        MergeResult with merged prompt and metadata
    """
    # Handle empty case
    if not modifications:
        return {
            "merged_prompt": original_prompt,
            "had_conflicts": False,
            "conflicts_resolved": 0,
            "merge_method": "no_modifications",
            "modifications_merged": 0
        }

    # Handle single modification case
    if len(modifications) == 1:
        return {
            "merged_prompt": modifications[0]["modified_prompt"],
            "had_conflicts": False,
            "conflicts_resolved": 0,
            "merge_method": "single_modification",
            "modifications_merged": 1
        }

    # Handle multiple modifications
    # Start with the first modification
    current_merged = modifications[0]["modified_prompt"]
    conflicts_resolved = 0
    had_conflicts = False

    # Try to merge each subsequent modification
    for i in range(1, len(modifications)):
        mod_current = modifications[i]

        # Check if changes are large
        diff_size = calculate_diff_size(original_prompt, mod_current["modified_prompt"])

        # Try automatic merge first
        automatic_merge = apply_non_conflicting_merge(
            original_prompt,
            current_merged,
            mod_current["modified_prompt"]
        )

        # If automatic merge succeeded and changes aren't too large
        if automatic_merge and diff_size < LARGE_CHANGE_THRESHOLD:
            current_merged = automatic_merge
        else:
            # Use LLM to merge
            had_conflicts = True

            # Create a temporary AgentModification for current merged state
            temp_mod: AgentModification = {
                "criterion": "merged",
                "original_prompt": original_prompt,
                "modified_prompt": current_merged,
                "changes_made": ["Previous modifications"],
                "reasoning": "Combined modifications",
                "tools_added": None,
                "mcp_servers_added": None
            }

            llm_merged = await llm_merge_modifications(
                original_prompt,
                temp_mod,
                mod_current,
                model,
                temperature
            )

            if llm_merged:
                current_merged = llm_merged
                conflicts_resolved += 1
            else:
                # If LLM merge fails, just use the latest modification
                print(f"Warning: LLM merge failed for modification {i}, using latest")
                current_merged = mod_current["modified_prompt"]

    # Determine merge method
    if len(modifications) == 1:
        merge_method = "single_modification"
    elif not had_conflicts:
        merge_method = "automatic"
    else:
        merge_method = "llm_assisted"

    return {
        "merged_prompt": current_merged,
        "had_conflicts": had_conflicts,
        "conflicts_resolved": conflicts_resolved,
        "merge_method": merge_method,
        "modifications_merged": len(modifications)
    }


async def merge_simulation_result(
    simulation_result: SimulationResult,
    original_prompt: str,
    **kwargs
) -> MergeResult:
    """Convenience function to merge a SimulationResult directly.

    Args:
        simulation_result: Result from simulation layer
        original_prompt: Original agent prompt
        **kwargs: Additional arguments passed to merge_modifications

    Returns:
        MergeResult with merged prompt
    """
    return await merge_modifications(
        modifications=simulation_result["modifications"],
        original_prompt=original_prompt,
        **kwargs
    )
