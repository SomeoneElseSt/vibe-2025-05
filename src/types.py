"""Type definitions for the agent orchestration system."""

from typing import TypedDict, List, Optional


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
