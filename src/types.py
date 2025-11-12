"""Type definitions for the agent orchestration system."""

from typing import TypedDict, List, Optional, Dict


class Message(TypedDict):
    """A single message in a conversation."""
    role: str
    content: str


class Conversation(TypedDict):
    """A complete conversation between agents."""
    messages: List[Message]
    metadata: dict


class ConversationResult(TypedDict):
    """Result from generating multiple conversations."""
    conversations: List[Conversation]
    base_agent_config: dict
    conversational_agent_prompts: List[str]


class CriterionScore(TypedDict):
    """Score for a single judging criterion."""
    criterion: str
    met: bool
    reasoning: Optional[str]


class ConversationJudgment(TypedDict):
    """Judgment for a single conversation."""
    conversation_index: int
    criteria_scores: List[CriterionScore]
    overall_pass: bool


class JudgmentResult(TypedDict):
    """Complete judgment result for all conversations."""
    judgments: List[ConversationJudgment]
    overall_statistics: Dict[str, any]
    judge_config: dict


class FixerTask(TypedDict):
    """Information about a single failed criterion that needs fixing."""
    criterion: str
    reasoning: Optional[str]
    conversation: Conversation
    conversation_index: int


class AgentModification(TypedDict):
    """Modification suggested by a fixer agent."""
    criterion: str
    original_prompt: str
    modified_prompt: str
    changes_made: List[str]
    reasoning: str
    tools_added: Optional[List[str]]
    mcp_servers_added: Optional[List[str]]


class SimulationResult(TypedDict):
    """Result from simulation layer."""
    modifications: List[AgentModification]
    total_issues: int
    total_fixed: int
    status: str


class IterationResult(TypedDict):
    """Result from a single orchestration iteration."""
    iteration: int
    prompt: str
    all_criteria_passed: bool
    total_conversations: int
    total_passed: int
    modification_applied: Optional[AgentModification]
    judgment_result: JudgmentResult


class OrchestrationResult(TypedDict):
    """Result from full orchestration loop."""
    success: bool
    final_prompt: str
    final_agent_file: Optional[str]  # Path to final agent file
    all_criteria_passed: bool
    iterations: List[IterationResult]
    total_iterations: int
    status: str


class MergeConflict(TypedDict):
    """Information about a merge conflict."""
    conflict_type: str  # "overlap", "large_change", "complex"
    original_text: str
    modification_1: str
    modification_2: str
    resolved_text: Optional[str]


class MergeResult(TypedDict):
    """Result from merging multiple modifications."""
    merged_prompt: str
    had_conflicts: bool
    conflicts_resolved: int
    merge_method: str  # "automatic", "llm_assisted", "single_modification"
    modifications_merged: int
