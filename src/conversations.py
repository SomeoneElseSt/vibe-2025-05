"""Conversation generation module for agent orchestration.

This module provides primitives for creating conversations between a base agent
and multiple conversational agents with different prompts.
"""

import asyncio
import os
import sys
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from dedalus_labs import AsyncDedalus, DedalusRunner

from .types import Message, Conversation, ConversationResult
from .config import DEFAULT_MAX_TURNS, DEFAULT_TEMPERATURE, CONVERSATION_MODEL

# Load environment variables
load_dotenv()


async def create_conversation(
    base_agent_prompt: str,
    conversational_agent_prompt: str,
    initial_message: str,
    model: str = CONVERSATION_MODEL,
    max_turns: int = DEFAULT_MAX_TURNS,
    temperature: float = DEFAULT_TEMPERATURE,
    tools: Optional[List] = None,
    mcp_servers: Optional[List[str]] = None,
) -> Optional[Conversation]:
    """Create a single conversation between two agents.

    Args:
        base_agent_prompt: System prompt for the base agent
        conversational_agent_prompt: System prompt for the conversational agent
        initial_message: Starting message for the conversation
        model: Model identifier to use
        max_turns: Maximum number of conversation turns
        temperature: Sampling temperature
        tools: Optional list of tool functions
        mcp_servers: Optional list of MCP server identifiers

    Returns:
        Conversation object or None if failed
    """
    client = AsyncDedalus()
    messages: List[Message] = []

    # Start with initial message from user - base agent should respond first
    current_message = initial_message
    current_role = "base_agent"  # Base agent responds to initial user message

    print(f"\n{'='*60}", file=sys.stderr, flush=True)
    print(f"STARTING CONVERSATION", file=sys.stderr, flush=True)
    print(f"{'='*60}", file=sys.stderr, flush=True)
    print(f"Initial message: {initial_message}", file=sys.stderr, flush=True)
    print(f"Max turns: {max_turns}", file=sys.stderr, flush=True)

    try:
        for turn in range(max_turns):
            print(f"\n--- Turn {turn + 1}/{max_turns} ({current_role}) ---", file=sys.stderr, flush=True)
            # Determine which agent speaks
            if current_role == "conversational_agent":
                system_prompt = conversational_agent_prompt
                next_role = "base_agent"
            else:
                system_prompt = base_agent_prompt
                next_role = "conversational_agent"

            # Create runner for current agent
            runner = DedalusRunner(client)

            # Build message history for context
            conversation_context = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": current_message}
            ]

            # Add previous messages as context (alternate user/assistant)
            if messages:
                for i, msg in enumerate(messages):
                    role_in_context = "assistant" if msg["role"] == current_role else "user"
                    conversation_context.insert(-1, {
                        "role": role_in_context,
                        "content": msg["content"]
                    })

            # Run the agent
            result = await runner.run(
                input=current_message,
                messages=conversation_context[:-1] if len(conversation_context) > 2 else None,
                model=model,
                temperature=temperature,
                max_steps=5,
                tools=tools or [],
                mcp_servers=mcp_servers or [],
                stream=False
            )

            # Extract response
            response = result.final_output
            if not response:
                return None

            # Store message
            messages.append({
                "role": current_role,
                "content": response
            })

            print(f"Response: {response[:200]}{'...' if len(response) > 200 else ''}", file=sys.stderr, flush=True)

            # Prepare for next turn
            current_message = response
            current_role = next_role

        conversation = {
            "messages": messages,
            "metadata": {
                "turns": len(messages),
                "model": model,
                "temperature": temperature,
                "base_agent_prompt": base_agent_prompt,
                "conversational_agent_prompt": conversational_agent_prompt,
                "initial_message": initial_message
            }
        }
        
        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print(f"CONVERSATION COMPLETE - {len(messages)} messages", file=sys.stderr, flush=True)
        print(f"{'='*60}\n", file=sys.stderr, flush=True)
        
        return conversation

    except Exception as e:
        print(f"Error creating conversation: {e}", file=sys.stderr, flush=True)
        return None


async def create_multiple_conversations(
    base_agent_prompt: str,
    conversational_agent_prompts: List[str],
    initial_message: str,
    model: str = CONVERSATION_MODEL,
    max_turns: int = DEFAULT_MAX_TURNS,
    temperature: float = DEFAULT_TEMPERATURE,
    tools: Optional[List] = None,
    mcp_servers: Optional[List[str]] = None,
    parallel: bool = True
) -> ConversationResult:
    """Create multiple conversations between base agent and different conversational agents.

    This is the main primitive for generating conversations. It creates one conversation
    for each conversational agent prompt provided.

    Args:
        base_agent_prompt: System prompt for the base agent
        conversational_agent_prompts: List of system prompts for conversational agents
        initial_message: Starting message for all conversations
        model: Model identifier to use
        max_turns: Maximum number of conversation turns per conversation
        temperature: Sampling temperature
        tools: Optional list of tool functions for agents
        mcp_servers: Optional list of MCP server identifiers
        parallel: Whether to run conversations in parallel (default True)

    Returns:
        ConversationResult containing all conversations and metadata
    """
    conversations: List[Conversation] = []

    if parallel:
        # Run all conversations concurrently
        tasks = [
            create_conversation(
                base_agent_prompt=base_agent_prompt,
                conversational_agent_prompt=conv_prompt,
                initial_message=initial_message,
                model=model,
                max_turns=max_turns,
                temperature=temperature,
                tools=tools,
                mcp_servers=mcp_servers
            )
            for conv_prompt in conversational_agent_prompts
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None and exceptions
        for result in results:
            if isinstance(result, Exception):
                print(f"Conversation failed with exception: {result}", file=sys.stderr)
                continue
            if result is not None:
                conversations.append(result)
    else:
        # Run conversations sequentially
        for conv_prompt in conversational_agent_prompts:
            result = await create_conversation(
                base_agent_prompt=base_agent_prompt,
                conversational_agent_prompt=conv_prompt,
                initial_message=initial_message,
                model=model,
                max_turns=max_turns,
                temperature=temperature,
                tools=tools,
                mcp_servers=mcp_servers
            )
            if result is not None:
                conversations.append(result)

    return {
        "conversations": conversations,
        "base_agent_config": {
            "prompt": base_agent_prompt,
            "model": model,
            "temperature": temperature,
            "tools": tools or [],
            "mcp_servers": mcp_servers or []
        },
        "conversational_agent_prompts": conversational_agent_prompts
    }


# Synchronous wrapper for convenience
def create_conversations_sync(
    base_agent_prompt: str,
    conversational_agent_prompts: List[str],
    initial_message: str,
    **kwargs
) -> ConversationResult:
    """Synchronous wrapper for create_multiple_conversations.

    Args:
        base_agent_prompt: System prompt for the base agent
        conversational_agent_prompts: List of system prompts for conversational agents
        initial_message: Starting message for all conversations
        **kwargs: Additional arguments passed to create_multiple_conversations

    Returns:
        ConversationResult containing all conversations and metadata
    """
    return asyncio.run(create_multiple_conversations(
        base_agent_prompt=base_agent_prompt,
        conversational_agent_prompts=conversational_agent_prompts,
        initial_message=initial_message,
        **kwargs
    ))
