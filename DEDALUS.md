# Dedalus Labs - Agents SDK Documentation

## Table of Contents
1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Installation & Setup](#installation--setup)
4. [Basic API Usage](#basic-api-usage)
5. [DedalusRunner - Agent Orchestration](#dedalusrunner---agent-orchestration)
6. [Multi-Model Routing](#multi-model-routing)
7. [MCP Integration](#mcp-integration)
8. [Policy System](#policy-system)
9. [Additional Capabilities](#additional-capabilities)
10. [Error Handling](#error-handling)
11. [Advanced Configuration](#advanced-configuration)
12. [Examples & Use Cases](#examples--use-cases)
13. [Resources](#resources)

---

## Overview

**Dedalus Labs** provides a unified API client library that simplifies building complex AI agents. It serves as a "Vercel for Agents" - a fully managed, production-grade platform that unifies the fragmented AI agent ecosystem into a single, drop-in API.

### Key Value Propositions

- **Universal Model Access**: Switch between GPT-4, Claude, Gemini, or any leading model with a single line of code
- **Hybrid Tooling**: Seamlessly combine your own Python functions with a rich marketplace of pre-built MCP tools
- **Multi-Model AI**: Orchestrate diverse LLMs natively with intelligent routing
- **Serverless Scale**: Deploy instantly, scale infinitely, no DevOps required
- **Production Ready**: Hot-reload logic & tools with zero downtime

### What Dedalus Provides

1. **Unified API Gateway**: Single interface for multiple AI model providers (OpenAI, Anthropic, Google, xAI, Mistral)
2. **Agent Orchestration**: `DedalusRunner` class for multi-turn conversations with automatic tool execution
3. **MCP Integration**: Connect to Model Context Protocol servers from marketplace or deploy your own
4. **Intelligent Routing**: Multi-model routing with handoffs, failover, and load balancing
5. **Policy System**: Fine-grained control over agent behavior at each step

---

## Core Concepts

### 1. Unified API Client

The Dedalus SDK provides both synchronous and asynchronous clients that work as drop-in replacements for OpenAI-compatible APIs, but with access to multiple providers.

### 2. DedalusRunner

A high-level agent orchestration system that:
- Manages conversation state across multiple turns
- Automatically executes Python functions as tools based on model requests
- Implements retry logic with configurable maximum steps
- Introspects Python functions to generate OpenAI-compatible tool schemas
- Supports both synchronous and asynchronous tool functions

### 3. Model Context Protocol (MCP)

MCP servers extend agent capabilities with external context sources:
- Marketplace of pre-built MCP servers (Slack, GitHub, web search, etc.)
- Deploy your own MCP servers in 3 clicks
- Mix local Python tools with remote MCP tools seamlessly

### 4. Multi-Model Routing

Intelligent model selection and handoffs:
- Automatic failover and load balancing
- Policy-based routing decisions
- Model handoffs for workflows requiring different capabilities
- Agent and model attributes for semantic routing

---

## Installation & Setup

### Python Installation

```bash
pip install dedalus-labs
```

### TypeScript Installation

```bash
npm install dedalus-labs
```

### Environment Setup

```python
import os
from dedalus_labs import Dedalus

# Option 1: Use DEDALUS_API_KEY environment variable
client = Dedalus()  # Automatically reads DEDALUS_API_KEY

# Option 2: Pass API key directly
client = Dedalus(api_key="your-dedalus-api-key")

# Option 3: Bring Your Own Key (BYOK) - Use provider keys directly
client = Dedalus(api_key=os.getenv("OPENAI_API_KEY"))  # Uses OpenAI key
client = Dedalus(api_key=os.getenv("ANTHROPIC_API_KEY"))  # Uses Anthropic key
```

### Getting an API Key

1. Log into your Dedalus dashboard
2. Navigate to the "API Keys" section
3. Generate a new API key

**Note**: With a `DEDALUS_API_KEY` in your environment, Dedalus handles routing to any provider or model for you, including handoffs between models from different providers.

---

## Basic API Usage

### Chat Completions

#### Basic Chat Completion

```python
from dedalus_labs import Dedalus
import os

client = Dedalus(api_key=os.environ.get("DEDALUS_API_KEY"))

completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    model="openai/gpt-4",
    temperature=0.7,
    max_tokens=500
)

print(f"Response: {completion.choices[0].message.content}")
print(f"Tokens used: {completion.usage.total_tokens}")
print(f"Model: {completion.model}")
```

#### Streaming Chat Completion

```python
stream = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "Write a short story about AI."}
    ],
    model="anthropic/claude-3-5-sonnet",
    stream=True,
    temperature=0.8
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

#### Multi-Model Routing (Automatic Failover)

```python
completion = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}],
    model=["openai/gpt-4", "anthropic/claude-3-5-sonnet"],  # List of models
    temperature=0.5
)
```

#### Tool Calling with Functions

```python
completion = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "What's the weather in San Francisco?"}
    ],
    model="openai/gpt-4",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }
    ],
    tool_choice="auto"
)

if completion.choices[0].message.tool_calls:
    for tool_call in completion.choices[0].message.tool_calls:
        print(f"Tool: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")
```

### Model Listing

```python
# List all available models
models_response = client.models.list()

print(f"Total models available: {len(models_response.data)}")

# Filter by provider
providers = {}
for model in models_response.data:
    provider = model.owned_by
    if provider not in providers:
        providers[provider] = []
    providers[provider].append(model.id)

for provider, model_ids in providers.items():
    print(f"\n{provider} ({len(model_ids)} models):")
    for model_id in model_ids[:5]:
        print(f"  - {model_id}")

# Get details for specific model
model = client.models.retrieve("openai/gpt-4")
print(f"Model ID: {model.id}")
print(f"Owner: {model.owned_by}")
print(f"Created: {model.created}")
```

### Embeddings

```python
from dedalus_labs import Dedalus
import numpy as np

client = Dedalus()

# Single text embedding
response = client.embeddings.create(
    input="The quick brown fox jumps over the lazy dog",
    model="text-embedding-3-small",
    encoding_format="float"
)

embedding = response.data[0].embedding
print(f"Embedding dimensions: {len(embedding)}")

# Batch embedding
texts = [
    "Machine learning is transforming industries.",
    "Artificial intelligence enables automation.",
    "Neural networks learn from data patterns."
]

response = client.embeddings.create(
    input=texts,
    model="text-embedding-3-large",
    dimensions=1024
)

embeddings_matrix = np.array([item.embedding for item in response.data])

# Calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarity = cosine_similarity(embeddings_matrix[0], embeddings_matrix[1])
print(f"Similarity: {similarity:.4f}")
```

### Audio Transcription

```python
from pathlib import Path
from dedalus_labs import Dedalus

client = Dedalus()

# Basic transcription
with open("audio.mp3", "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        file=audio_file,
        model="openai/whisper-1",
        response_format="text"
    )

print(f"Transcription: {transcription}")

# Transcription with language hint and verbose output
transcription = client.audio.transcriptions.create(
    file=Path("/path/to/meeting.mp3"),
    model="openai/whisper-1",
    language="en",
    prompt="This is a business meeting about Q4 sales targets.",
    response_format="verbose_json",
    temperature=0.2
)

print(f"Text: {transcription.text}")
print(f"Language: {transcription.language}")
print(f"Duration: {transcription.duration}s")

# Generate SRT subtitles
srt_output = client.audio.transcriptions.create(
    file=Path("/path/to/video.mp4"),
    model="openai/whisper-1",
    response_format="srt"
)

with open("subtitles.srt", "w") as f:
    f.write(srt_output)
```

### Audio Translation

```python
# Translate foreign language audio to English
translation = client.audio.translations.create(
    file=Path("/path/to/spanish_audio.mp3"),
    model="openai/whisper-1",
    response_format="text"
)

print(f"English translation: {translation}")
```

### Text-to-Speech

```python
# Basic speech synthesis
response = client.audio.speech.create(
    input="Hello! This is a test of the text to speech system.",
    model="openai/tts-1",
    voice="alloy",
    response_format="mp3"
)

output_path = Path("output.mp3")
response.write_to_file(output_path)

# High-quality speech with custom settings
speech = client.audio.speech.create(
    input="Welcome to our podcast. Today we'll discuss AI developments.",
    model="openai/tts-1-hd",  # Higher quality
    voice="nova",  # Available: alloy, ash, ballad, coral, echo, fable, onyx, nova, sage, shimmer, verse
    speed=1.1,  # 0.25 to 4.0
    response_format="opus"
)

speech.write_to_file("podcast_intro.opus")
```

### Image Generation

```python
from pathlib import Path
from dedalus_labs import Dedalus
import base64

client = Dedalus()

# Basic image generation
response = client.images.generate(
    prompt="A serene mountain landscape at sunset with a lake reflection",
    model="openai/dall-e-3",
    size="1024x1024",
    quality="standard",
    n=1
)

image_url = response.data[0].url
print(f"Generated image URL: {image_url}")

# Generate as base64
response = client.images.generate(
    prompt="Abstract geometric patterns in blue and gold",
    model="openai/dall-e-3",
    size="1792x1024",
    quality="hd",
    style="vivid",  # or "natural"
    response_format="b64_json"
)

image_data = base64.b64decode(response.data[0].b64_json)
with open("generated_image.png", "wb") as f:
    f.write(image_data)
```

### Image Editing

```python
# Basic image editing
response = client.images.edit(
    image=Path("/path/to/original.png"),
    prompt="Add a sunset sky with vibrant orange and purple colors",
    model="openai/dall-e-2",
    size="1024x1024",
    n=1
)

print(f"Edited image URL: {response.data[0].url}")

# Image editing with mask
with open("photo.png", "rb") as image_file, open("mask.png", "rb") as mask_file:
    response = client.images.edit(
        image=image_file,
        mask=mask_file,  # White areas will be regenerated
        prompt="Replace the background with a tropical beach scene",
        model="openai/dall-e-2",
        n=2,
        size="512x512"
    )
```

### Health Check

```python
from dedalus_labs import Dedalus
import dedalus_labs

client = Dedalus()

try:
    health = client.health.check()
    print(f"API Status: {health.status}")
except dedalus_labs.APIConnectionError as e:
    print(f"Cannot reach API server: {e}")
except dedalus_labs.APIStatusError as e:
    print(f"API returned error status: {e.status_code}")
```

### Async Client Usage

```python
import asyncio
from dedalus_labs import AsyncDedalus

async def main():
    async with AsyncDedalus() as client:
        # Async chat completion
        completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello!"}],
            model="openai/gpt-4"
        )
        print(f"Response: {completion.choices[0].message.content}")

        # Async streaming
        stream = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Tell me a story"}],
            model="anthropic/claude-3-5-sonnet",
            stream=True
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()

        # Concurrent API calls
        tasks = [
            client.embeddings.create(
                input=f"Sample text {i}",
                model="text-embedding-3-small"
            )
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        print(f"Processed {len([r for r in results if not isinstance(r, Exception)])} embeddings")

asyncio.run(main())
```

---

## DedalusRunner - Agent Orchestration

`DedalusRunner` is the core agent orchestration system that enables multi-turn conversations with automatic tool execution, intelligent model routing, and policy-based execution control.

### Basic Usage

```python
from dedalus_labs import Dedalus, DedalusRunner
import json

# Define tools as Python functions
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get current weather for a location"""
    weather_data = {
        "San Francisco": {"temp": 18, "condition": "Cloudy"},
        "New York": {"temp": 22, "condition": "Sunny"},
        "London": {"temp": 15, "condition": "Rainy"}
    }
    data = weather_data.get(location, {"temp": 20, "condition": "Unknown"})
    return json.dumps({
        "location": location,
        "temperature": data["temp"],
        "unit": unit,
        "condition": data["condition"]
    })

def search_database(query: str) -> str:
    """Search internal database"""
    results = ["Result 1", "Result 2", "Result 3"]
    return json.dumps({"query": query, "results": results, "count": len(results)})

# Initialize runner
client = Dedalus()
runner = DedalusRunner(client)

# Basic run with automatic tool execution
result = runner.run(
    input="What's the weather like in San Francisco and New York?",
    tools=[get_weather, search_database],
    model="openai/gpt-4",
    max_steps=10,
    temperature=0.7
)

print(f"Final Answer: {result.final_output}")
print(f"Tools Called: {result.tools_called}")
print(f"Steps Used: {result.steps_used}/{result.max_steps}")
```

### Multi-Turn Conversations

```python
# First turn
result1 = runner.run(
    input="Calculate the weather difference between London and New York",
    tools=[get_weather],
    model="openai/gpt-4",
    max_steps=5
)

print(f"Turn 1: {result1.final_output}")

# Continue conversation with history
result2 = runner.run(
    input="Which city is warmer?",
    messages=result1.messages,  # Pass previous context
    tools=[get_weather],
    model="openai/gpt-4",
    max_steps=5
)

print(f"Turn 2: {result2.final_output}")
```

### Streaming Mode

```python
result = runner.run(
    input="Tell me about the weather and then search for climate data",
    tools=[get_weather, search_database],
    model="anthropic/claude-3-5-sonnet",
    max_steps=10,
    stream=True,
    verbose=True  # Print debug information
)

# Stream tokens as they arrive
for token in result:
    print(token, end="", flush=True)
print()
```

### Async Tools Support

```python
import asyncio

async def async_api_call(endpoint: str, method: str = "GET") -> dict:
    """Call an external API asynchronously"""
    await asyncio.sleep(0.1)  # Simulate network delay
    return {
        "endpoint": endpoint,
        "method": method,
        "status": 200,
        "data": {"key": "value"}
    }

# Runner supports both sync and async tools
result = runner.run(
    input="Fetch data from /api/users endpoint",
    tools=[async_api_call],
    model="openai/gpt-4",
    max_steps=5
)
print(result.final_output)
```

### Runner Parameters

- `input` (str): The user's input/prompt
- `model` (str | List[str]): Model identifier(s) - can be single model or list for routing
- `tools` (List[Callable]): Python functions to use as tools
- `mcp_servers` (List[str]): MCP server identifiers from marketplace
- `max_steps` (int): Maximum number of steps/conversation turns
- `temperature` (float): Sampling temperature
- `stream` (bool): Enable streaming output
- `verbose` (bool): Print debug information
- `messages` (List[dict]): Previous conversation history
- `policy` (Callable): Custom policy function for step-by-step control
- `agent_attributes` (dict): Semantic metadata for routing
- `model_attributes` (dict): Model-specific attributes
- `handoff_config` (dict): Configuration for model handoffs
- `on_tool_event` (Callable): Callback for tool execution events

### Runner Response Object

The `result` object contains:
- `final_output` (str): The final response text
- `tools_called` (List[str]): List of tool names that were called
- `steps_used` (int): Number of steps actually used
- `max_steps` (int): Maximum steps configured
- `messages` (List[dict]): Full conversation history

---

## Multi-Model Routing

Dedalus supports intelligent routing between multiple models with automatic failover, load balancing, and handoffs.

### Simple Multi-Model Routing

```python
result = runner.run(
    input="Solve this complex math problem: integrate x^2 * sin(x) dx",
    model=["openai/gpt-4", "anthropic/claude-3-5-sonnet", "google/gemini-pro"],
    max_steps=10
)

print(f"Solution: {result.final_output}")
print(f"Model used: {result.messages[-1].get('model', 'unknown')}")
```

### Policy-Based Routing

```python
def smart_router(context):
    """Route to different models based on conversation state"""
    step = context["step"]
    messages = context["messages"]

    # Use faster model for initial processing
    if step <= 2:
        return {"model": "openai/gpt-3.5-turbo", "temperature": 0.5}

    # Switch to stronger model for complex reasoning
    if any("error" in msg.get("content", "").lower() for msg in messages[-3:]):
        return {"model": "anthropic/claude-3-5-sonnet", "temperature": 0.3}

    # Default to GPT-4
    return {"model": "openai/gpt-4", "temperature": 0.7}

result = runner.run(
    input="Analyze this data and provide insights",
    model=["openai/gpt-3.5-turbo", "openai/gpt-4", "anthropic/claude-3-5-sonnet"],
    policy=smart_router,
    max_steps=15
)
```

### Agent and Model Attributes

```python
result = runner.run(
    input="Write creative marketing copy for a tech product",
    model=["openai/gpt-4", "anthropic/claude-3-5-sonnet"],
    agent_attributes={
        "creativity": 0.9,
        "formality": 0.3,
        "brevity": 0.6
    },
    model_attributes={
        "openai/gpt-4": {"speed": 0.7, "quality": 0.9},
        "anthropic/claude-3-5-sonnet": {"speed": 0.8, "quality": 0.95}
    },
    temperature=0.8,
    max_steps=10
)
```

### Handoff Configuration

```python
result = runner.run(
    input="Research and write a technical report on quantum computing",
    model=["openai/gpt-4", "anthropic/claude-3-5-sonnet"],
    handoff_config={
        "strategy": "quality_based",
        "threshold": 0.7,
        "max_handoffs": 3
    },
    max_steps=20
)
```

---

## MCP Integration

Model Context Protocol (MCP) servers extend agent capabilities with external context sources.

### Basic MCP Usage

```python
result = runner.run(
    input="Search our Slack for messages about the Q4 roadmap and summarize",
    mcp_servers=["slack"],  # Enable Slack MCP server
    model="openai/gpt-4",
    max_steps=10
)

print(f"Summary: {result.final_output}")
```

### Multiple MCP Servers

```python
result = runner.run(
    input="Check GitHub issues and Slack discussions about bug #1234",
    mcp_servers=["slack", "github"],
    model="anthropic/claude-3-5-sonnet",
    max_steps=15,
    verbose=True
)

print(f"Analysis: {result.final_output}")
print(f"MCP tools used: {[t for t in result.tools_called if 'mcp_' in t]}")
```

### Combining MCP with Custom Tools

```python
def calculate_metrics(data: str) -> str:
    """Calculate custom metrics"""
    return json.dumps({"metric": "engagement", "value": 0.85})

result = runner.run(
    input="Analyze our GitHub repository stats and calculate engagement metrics",
    tools=[calculate_metrics],
    mcp_servers=["github"],
    model="openai/gpt-4",
    max_steps=10
)
```

### MCP Server Marketplace

Dedalus provides a marketplace of pre-built MCP servers:
- **Web Search**: `tsion/brave-search-mcp`
- **Slack**: `slack`
- **GitHub**: `github`
- **PDF Reader**: `dedalus/advanced-pdf-reader-v2`
- And many more...

### Deploying Your Own MCP Server

1. Upload your MCP server from GitHub repo
2. Dedalus hosts it for free
3. Use it with a simple slug identifier
4. No Docker files or YAML configuration needed

### MCP Server Guidelines

- Servers should be stateless (authentication coming soon)
- Use Streamable HTTP transport for production
- Follow MCP protocol specifications
- See: https://docs.dedaluslabs.ai/server-guidelines.md

---

## Policy System

Policies provide fine-grained control over agent behavior at each step of execution.

### Basic Policy

```python
def custom_policy(context: dict) -> dict:
    """Custom policy function"""
    step = context.get("step", 1)
    policy = {}
    
    # Modify behavior at specific steps
    if step == 3:
        policy["message_prepend"] = [
            {"role": "system", "content": "You must speak like a pirate."}
        ]
    
    # Set maximum steps
    policy["max_steps"] = 4
    
    return policy

result = runner.run(
    input="Step 1) Add 7 and 8. Step 2) Multiply by 3. Step 3) Search the web.",
    model="openai/gpt-4.1",
    tools=[add, mul],
    mcp_servers=["tsion/brave-search-mcp"],
    policy=custom_policy
)
```

### Policy Context

The policy function receives a context dictionary with:
- `step` (int): Current step number
- `messages` (List[dict]): Conversation history
- `tools_called` (List[str]): Tools called so far
- `model` (str): Current model being used
- `agent_attributes` (dict): Agent metadata
- `model_attributes` (dict): Model metadata

### Policy Return Values

The policy function can return a dictionary with:
- `model` (str): Override model selection
- `temperature` (float): Override temperature
- `max_steps` (int): Override maximum steps
- `message_prepend` (List[dict]): Messages to prepend to conversation
- `message_append` (List[dict]): Messages to append to conversation
- `tool_filter` (List[str]): Filter available tools

### Advanced Policy Example

```python
def on_tool(evt: dict) -> None:
    print("[policy tool evt]", json.dumps(evt))

def policy(ctx: dict) -> dict:
    step = ctx.get("step", 1)
    print(f"[policy] step={step}")
    pol: dict = {}

    if step == 3:
        pol.update({
            "message_prepend": [
                {"role": "system", "content": "You must speak like a pirate."}
            ]
        })
    
    # Cap total steps for safety
    pol.setdefault("max_steps", 4)
    return pol

result = runner.run(
    input="...",
    model="openai/gpt-4.1",
    tools=[add, mul],
    mcp_servers=["tsion/brave-search-mcp"],
    stream=True,
    on_tool_event=on_tool,
    policy=policy
)
```

---

## Additional Capabilities

### Embeddings

See [Basic API Usage - Embeddings](#embeddings) section above.

### Audio Processing

- **Transcription**: Convert audio to text using Whisper
- **Translation**: Translate audio in any language to English
- **Text-to-Speech**: Generate natural-sounding speech

See [Basic API Usage - Audio](#audio-transcription) section above.

### Image Generation & Editing

- **Generation**: Create images from text using DALL-E
- **Editing**: Edit existing images with text prompts and masks
- **Variations**: Generate variations of existing images

See [Basic API Usage - Image Generation](#image-generation) section above.

---

## Error Handling

### Error Types

```python
from dedalus_labs import Dedalus
import dedalus_labs

client = Dedalus()

try:
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello"}],
        model="invalid/model"
    )
except dedalus_labs.NotFoundError as e:
    print(f"Model not found: {e.status_code}")
except dedalus_labs.AuthenticationError as e:
    print(f"Authentication failed: {e}")
    print("Check your DEDALUS_API_KEY environment variable")
except dedalus_labs.RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except dedalus_labs.BadRequestError as e:
    print(f"Invalid request: {e.status_code}")
    print(f"Response body: {e.body}")
except dedalus_labs.APIConnectionError as e:
    print(f"Connection failed: {e}")
except dedalus_labs.InternalServerError as e:
    print(f"Server error: {e.status_code}")
except dedalus_labs.APIError as e:
    print(f"Generic API error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Retry Configuration

```python
# Global retry configuration
client_with_retries = Dedalus(max_retries=5)

# Per-request retry override
completion = client.with_options(max_retries=0).chat.completions.create(
    messages=[{"role": "user", "content": "Test"}],
    model="openai/gpt-4"
)
```

### Timeout Configuration

```python
import httpx

client_with_timeout = Dedalus(
    timeout=httpx.Timeout(
        connect=5.0,  # Connection timeout
        read=30.0,    # Read timeout
        write=10.0,   # Write timeout
        pool=5.0      # Pool timeout
    )
)
```

### Raw Response Access

```python
response = client.chat.completions.with_raw_response.create(
    messages=[{"role": "user", "content": "Debug"}],
    model="openai/gpt-4"
)

print(f"Status: {response.http_response.status_code}")
print(f"Headers: {dict(response.http_response.headers)}")
print(f"Request ID: {response.http_response.headers.get('x-request-id')}")

completion = response.parse()
print(f"Parsed response: {completion.choices[0].message.content}")
```

---

## Advanced Configuration

### Custom Headers

```python
client = Dedalus(
    default_headers={
        "User-Agent": "MyApp/1.0",
        "X-Custom-Header": "custom-value"
    }
)
```

### Proxy Configuration

```python
import httpx
from dedalus_labs import Dedalus, DefaultHttpxClient

client = Dedalus(
    http_client=DefaultHttpxClient(
        proxy="http://proxy.example.com:8080",
        transport=httpx.HTTPTransport(
            local_address="0.0.0.0",
            retries=3
        )
    )
)
```

### Custom Base URL

```python
# For self-hosted deployments
client = Dedalus(
    base_url="https://api.mycompany.com",
    api_key="custom_key"
)

# Or use environment variable
import os
os.environ["DEDALUS_BASE_URL"] = "https://custom.api.com"
client = Dedalus()  # Will use custom base URL
```

### Development Environment

```python
client = Dedalus(
    environment="development",  # Uses http://localhost:8080
    api_key="dev_key"
)
```

### Context Manager

```python
with Dedalus() as client:
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello"}],
        model="openai/gpt-4"
    )
    print(completion.choices[0].message.content)
# HTTP connection automatically closed
```

### Per-Request Options Override

```python
base_client = Dedalus(timeout=60.0, max_retries=2)

# Override for specific request
completion = base_client.with_options(
    timeout=10.0,
    max_retries=0
).chat.completions.create(
    messages=[{"role": "user", "content": "Quick request"}],
    model="openai/gpt-3.5-turbo"
)
```

### Organization ID

```python
client = Dedalus(
    api_key="key",
    organization="org_123456"
)
```

---

## Examples & Use Cases

### Example 1: Hello World

```python
import asyncio
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv

load_dotenv()

async def main():
    client = AsyncDedalus()
    runner = DedalusRunner(client)

    response = await runner.run(
        input="What was the score of the 2025 Wimbledon final?",
        model="openai/gpt-4o-mini"
    )

    print(response.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 2: Basic Tools

```python
import asyncio
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv

load_dotenv()

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

async def main():
    client = AsyncDedalus()
    runner = DedalusRunner(client)

    result = await runner.run(
        input="Calculate (15 + 27) * 2", 
        model="openai/gpt-4.1", 
        tools=[add, multiply]
    )

    print(f"Result: {result.final_output}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 3: MCP Integration

```python
import asyncio
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv

load_dotenv()

async def main():
    client = AsyncDedalus()
    runner = DedalusRunner(client)

    result = await runner.run(
        input="Who won Wimbledon 2025?",
        model="openai/gpt-4o-mini",
        mcp_servers=["tsion/brave-search-mcp"],
        stream=False
    )

    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 4: Tool Chaining

```python
import asyncio
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv

load_dotenv()

def add(a: int, b: int) -> int:
    return a + b

def mul(x: int, y: int) -> int:
    return x * y

async def main():
    client = AsyncDedalus()
    runner = DedalusRunner(client)

    result = await runner.run(
        input=(
            "1. Add 2 and 3. "
            "2. Multiply that by 4. "
            "3. Multiply this number by the age of the winner of the 2025 Wimbledon men's singles final. "
            "Use your tools to do this."
        ),
        model=["openai/gpt-4.1"],
        tools=[add, mul],
        mcp_servers=["tsion/brave-search-mcp"],
        stream=False
    )

    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 5: Model Handoffs

```python
import os
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv

load_dotenv()

async def main():
    client = AsyncDedalus()
    runner = DedalusRunner(client)

    result = await runner.run(
        input="Find the year GPT-5 released, and handoff to Claude to write a haiku about Elon Musk.",
        model=["openai/gpt-4.1", "claude-3-5-sonnet-20241022"],
        mcp_servers=["tsion/brave-search-mcp"],
        stream=False
    )

    print(result.final_output)

if __name__ == "__main__":
    main()
```

### Example 6: Streaming

```python
import asyncio
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv
from dedalus_labs.utils.streaming import stream_async

load_dotenv()

async def main():
    client = AsyncDedalus()
    runner = DedalusRunner(client)

    result = runner.run(
        input="What do you think of Mulligan?",
        model="openai/gpt-4o-mini",
        stream=True
    )

    # Use stream parameter and stream_async function to stream output
    await stream_async(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 7: Custom Policy

```python
import asyncio
import json
from dedalus_labs import AsyncDedalus, DedalusRunner
from dedalus_labs.utils.streaming import stream_async
from dotenv import load_dotenv

load_dotenv()

def on_tool(evt: dict) -> None:
    print("[policy tool evt]", json.dumps(evt))

def add(a: int, b: int) -> int:
    print(f"[tool:add] a={a} b={b}")
    return a + b

def mul(x: int, y: int) -> int:
    print(f"[tool:mul] x={x} y={y}")
    return x * y

def policy(ctx: dict) -> dict:
    step = ctx.get("step", 1)
    print(f"[policy] step={step}")
    pol: dict = {}

    if step == 3:
        pol.update({
            "message_prepend": [
                {"role": "system", "content": "You must speak like a pirate."}
            ]
        })
    
    # Cap total steps for safety
    pol.setdefault("max_steps", 4)
    return pol

async def main() -> None:
    client = AsyncDedalus()
    runner = DedalusRunner(client)

    prompt = (
        "Step 1) Add 7 and 8. "
        "Step 2) Multiply the previous result by 3. "
        "Step 3) Search the web for the final number and summarize the first result title. "
    )

    result = runner.run(
        input=prompt,
        model="openai/gpt-4.1",
        tools=[add, mul],
        mcp_servers=["tsion/brave-search-mcp"],
        stream=True,
        on_tool_event=on_tool,
        policy=policy
    )

    await stream_async(result)    

if __name__ == "__main__":
    asyncio.run(main())
```

### Use Case: Legal AI Agent

```python
import dedalus as ddls

def fetch_case_files(case_id: str):
    # ... your database logic ...
    return {"files": [...]}

def store_analysis(report: dict):
    # ... your storage logic ...
    return {"status": "success"}

paralegal = ddls.create_agent(
    models=[
        "anthropic/claude-3-5-sonnet-20241022",  # For deep analysis
        "groq/llama-3.3-70B-specdec",  # For fast Q&A
    ],
    tools=[
        fetch_case_files,
        store_analysis,
        "dedalus/advanced-pdf-reader-v2",
        "mcp/legal-citation-checker"
    ],
    instructions="You are a specialized legal AI..."
)

response = paralegal.chat.completions.create(
    messages=[{"role": "user", "content": "Analyze case #1234"}]
)
```

### Use Case: Financial Analyst Agent

```python
import dedalus as ddls

def get_portfolio_data(user_id: str):
    # ... your logic ...
    return {"portfolio": [...]}

def execute_trade_order(order: dict):
    # ... your logic ...
    return {"status": "executed"}

analyst = ddls.create_agent(
    models=[
        "cohere/command-r-plus",  # Semantic search over data
        "your-org/secure-finance-llm-v1",  # Personalized advice (private)
        "openai/gpt-4.1-mini",  # General tasks, cost-effective
    ],
    tools=[
        get_portfolio_data,
        execute_trade_order,
        "mcp/market-data-api-v3",
        "dedalus/risk-assessment-tool"
    ],
    instructions="You deliver tailored financial insights..."
)
```

### Use Case: Code Architect Agent

```python
import dedalus as ddls

def traverse_project_files():
    # ... your logic ...
    return {"files": [...]}

def edit_file(changes: dict):
    # ... your logic ...
    return {"status": "updated"}

architect = ddls.create_agent(
    models=[
        "google/gemini-1.5-pro-latest",  # Full codebase understanding
        "anthropic/claude-3-5-sonnet-20241022",  # Surgical code generation
    ],
    tools=[
        traverse_project_files,
        apply_code_modifications,
        "community/speech-to-text-whisper",
        "dedalus/static-code-analyzer"
    ],
    instructions="You are a codebase refactoring expert..."
)
```

---

## Resources

### Official Documentation

- **Main Documentation**: https://docs.dedaluslabs.ai/
- **Quickstart Guide**: https://docs.dedaluslabs.ai/quickstart.md
- **Examples**: https://docs.dedaluslabs.ai/examples/
- **FAQ**: https://docs.dedaluslabs.ai/faq
- **MCP Server Guidelines**: https://docs.dedaluslabs.ai/server-guidelines.md
- **BYOK Guide**: https://docs.dedaluslabs.ai/byok.md
- **llms.txt**: https://docs.dedaluslabs.ai/llms.txt (for AI assistants)

### GitHub Repositories

- **Python SDK**: https://github.com/dedalus-labs/dedalus-sdk-python
- **TypeScript SDK**: https://github.com/dedalus-labs/dedalus-sdk-typescript
- **Go SDK**: https://github.com/dedalus-labs/dedalus-sdk-go
- **OpenAPI Spec**: https://github.com/dedalus-labs/dedalus-openapi
- **Organization**: https://github.com/dedalus-labs

### Community & Support

- **Website**: https://www.dedaluslabs.ai
- **Discord**: Join the community Discord
- **Email Support**: support@dedaluslabs.ai

### Example MCP Servers

- **Brave Search MCP**: `tsion/brave-search-mcp`
- **Advanced PDF Reader**: `dedalus/advanced-pdf-reader-v2`
- **Legal Citation Checker**: `mcp/legal-citation-checker`
- **Market Data API**: `mcp/market-data-api-v3`
- **Static Code Analyzer**: `dedalus/static-code-analyzer`

### Model Identifiers

Common model formats:
- `openai/gpt-4`
- `openai/gpt-4o-mini`
- `openai/gpt-3.5-turbo`
- `anthropic/claude-3-5-sonnet-20241022`
- `anthropic/claude-3-5-sonnet`
- `google/gemini-pro`
- `google/gemini-1.5-pro-latest`
- `cohere/command-r-plus`
- `groq/llama-3.3-70B-specdec`

### Key Features Summary

1. **5 Lines of Code**: Build complex agents with minimal code
2. **Any Model, Any Tool**: Universal access to models and tools
3. **MCP Marketplace**: Discover and use production-ready tools
4. **Mix-and-Match**: Combine local Python functions with cloud-hosted MCP servers
5. **Hot Reloading**: Update agent logic and tools without downtime
6. **Multi-Model Routing**: Intelligent model selection and handoffs
7. **Policy System**: Fine-grained control over agent behavior
8. **Production Ready**: Serverless scale, error handling, retries, streaming

### Important Notes

- **Authentication**: Currently supports stateless servers. Authentication coming soon.
- **Languages**: Python (stable), TypeScript (beta), Go (available)
- **Pricing**: Transparent, dev-first pricing model
- **Marketplace Monetization**: 80% creator share, instant payouts (coming soon)
- **Open Source**: SDKs are MIT licensed
- **BYOK Support**: Use your own API keys from any supported provider

---

## Quick Reference

### Installation
```bash
pip install dedalus-labs
```

### Basic Client
```python
from dedalus_labs import Dedalus
client = Dedalus()
```

### Basic Runner
```python
from dedalus_labs import Dedalus, DedalusRunner
client = Dedalus()
runner = DedalusRunner(client)
result = runner.run(input="Hello", model="openai/gpt-4")
```

### With Tools
```python
def my_tool(arg: str) -> str:
    return f"Processed: {arg}"

result = runner.run(
    input="Use my tool",
    tools=[my_tool],
    model="openai/gpt-4"
)
```

### With MCP
```python
result = runner.run(
    input="Search the web",
    mcp_servers=["tsion/brave-search-mcp"],
    model="openai/gpt-4"
)
```

### Multi-Model
```python
result = runner.run(
    input="Complex task",
    model=["openai/gpt-4", "anthropic/claude-3-5-sonnet"],
    max_steps=10
)
```

### Streaming
```python
result = runner.run(
    input="Long response",
    model="openai/gpt-4",
    stream=True
)
for token in result:
    print(token, end="", flush=True)
```

---

*Last Updated: Based on research conducted via Perplexity MCP and Context7 MCP*
*Documentation Version: 1.0*
