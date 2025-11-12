"""Configuration and constants for the system."""

import os
from typing import Optional

# Model configuration
DEFAULT_MODEL = "openai/gpt-4o-mini"
CONVERSATION_MODEL = "openai/gpt-4o-mini"

# Conversation parameters
DEFAULT_MAX_TURNS = 10
DEFAULT_TEMPERATURE = 0.7

# Environment variables
def get_api_key() -> Optional[str]:
    """Get Dedalus API key from environment."""
    return os.getenv("DEDALUS_API_KEY")
