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
Available MCP Servers from Dedalus Marketplace:

SEARCH & DISCOVERY:
- brave-search-mcp - Integrate web search and local search capabilities with Brave
- sonar - Enable AI assistants to perform web searches using Perplexity's Sonar Pro
- exa-search-mcp - Fast, intelligent web search and crawling
- exa-mcp - Fast, intelligent web search and crawling

DATA & ANALYTICS:
- Historic-Data-MCP - Demand sentiment & price trend analysis
- yahoo-finance-mcp - Get real-time stock quotes, historical OHLCV data, news headlines, and multi-stock comparisons
- yclistdedalus - Comprehensive Y Combinator startup database with advanced search and analytics
- tweet-mcp - Y Combinator tweet search for startup insights and community intelligence

LOCATION & PLACES:
- foursquare-places-mcp - Search for places and place recommendations from the Foursquare Places database
- city-info-mcp - City information and data
- tabelog-mcp - Restaurant and dining information

ENTERTAINMENT & EVENTS:
- ticketmaster-mcp - Discover events, venues, and attractions through the Ticketmaster Discovery API
- mcp-games-public - Gaming and entertainment tools

PRODUCTIVITY & COLLABORATION:
- mcp-linear - Complete Linear project management integration for AI-powered workflow automation
- airtable-mcp-server - Airtable integration for database management
- notion-mcp - Essential Notion workspace integration for AI-powered page management
- agentmail-mcp - Toolset for managing email communications via the AgentMail API
- dedalusgmailmcp - Full Gmail automation for AI-powered email management and workflows

DEVELOPER TOOLS:
- context7 - Fetch up-to-date, version-specific documentation and code examples directly into your prompts
- sequential-thinking - Dynamic and reflective problem-solving through a structured thinking process
- hf-mcp - Hugging Face Hub integration for AI model and dataset discovery
- OnKernelDedalusMCP - Deploy and scale AI web agents
- mcp-server-example-python - Example MCP server implementation

UTILITIES:
- open-meteo-mcp - Weather and meteorological data
- general-mcp - Essential time and date utility for AI agents requiring temporal context
- ffmpeg-mcp - Media processing and conversion
- cat-facts - Fun facts and trivia
- simple-mcp - Simple utility server
- oscarmcp - Utility server
- civitai-mcp-server - AI model and content discovery
- scale-advisor - Claude AI integration for enhanced conversational capabilities
- arpamcp - Utility server
- HackPrinceton - Hackathon and event tools
- mcptest001 - Test server

Note: Some servers may have multiple versions from different authors. Use the specific server name/author format when referencing them.
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
