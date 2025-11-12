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
