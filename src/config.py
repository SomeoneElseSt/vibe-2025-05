"""Configuration and constants for the system."""

import os
from typing import Optional

# Model configuration
DEFAULT_MODEL = "openai/gpt-4o-mini"
CONVERSATION_MODEL = "openai/gpt-4o-mini"

# Conversation parameters
DEFAULT_MAX_TURNS = 10
DEFAULT_TEMPERATURE = 0.7

# Simulator configuration
FIXER_MODEL = "openai/gpt-4o-mini"
FIXER_TEMPERATURE = 0.3

# Available MCP servers with descriptions
AVAILABLE_MCP_SERVERS = """
Available MCP Servers:
1. tsion/brave-search-mcp - Web search using Brave Search API
2. slack - Access Slack messages and channels
3. github - Access GitHub issues, PRs, and repositories
4. dedalus/advanced-pdf-reader-v2 - Read and analyze PDF documents
5. mcp/legal-citation-checker - Check legal citations
6. mcp/market-data-api-v3 - Access market data and financial information
7. dedalus/static-code-analyzer - Analyze code for issues
8. community/speech-to-text-whisper - Convert speech to text
"""

# Instructions for adding tools
TOOL_ADDING_INSTRUCTIONS = """
How to Add Tools to Your Agent:

1. Define Python Functions:
   - Write a function with clear docstring
   - Use type hints for all parameters
   - Return serializable data (str, dict, list, etc.)

   Example:
   ```python
   def get_weather(location: str, unit: str = "celsius") -> str:
       \"\"\"Get current weather for a location.\"\"\"
       # Your implementation here
       return json.dumps({"temperature": 20, "condition": "sunny"})
   ```

2. Pass Tools to DedalusRunner:
   - Add your function to the tools list
   - The SDK automatically generates OpenAI-compatible schemas from function signatures

   Example:
   ```python
   result = runner.run(
       input="What's the weather?",
       tools=[get_weather, another_tool],
       model="openai/gpt-4o-mini"
   )
   ```

3. Add MCP Servers:
   - Simply add the server name to mcp_servers list

   Example:
   ```python
   result = runner.run(
       input="Search the web",
       mcp_servers=["tsion/brave-search-mcp"],
       model="openai/gpt-4o-mini"
   )
   ```

4. Combine Tools and MCP:
   ```python
   result = runner.run(
       input="Complex task",
       tools=[my_custom_tool],
       mcp_servers=["slack", "github"],
       model="openai/gpt-4o-mini"
   )
   ```
"""

# Environment variables
def get_api_key() -> Optional[str]:
    """Get Dedalus API key from environment."""
    return os.getenv("DEDALUS_API_KEY")
