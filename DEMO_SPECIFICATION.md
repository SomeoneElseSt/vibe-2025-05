# Demo Specification: Agent Orchestration Platform

## Overview
Build a beautiful, YC startup-styled frontend platform that demonstrates an AI agent improvement system. The platform visualizes how agents are automatically improved through conversation simulation, evaluation against SOPs (Standard Operating Procedures), and iterative fixes using parallel "fixer" agents.

## Tech Stack Requirements
- **Frontend**: Next.js with shadcn-ui components
- **Backend**: Python-based agent orchestration system (already implemented)
- **Styling**: Modern, clean YC startup aesthetic (think Vercel, Linear, or Stripe dashboard style)

## Demo Flow & User Experience

### Initial State
1. **Agent Selection View**
   - Display available agents in a card-based grid
   - Each agent card shows:
     - Agent name/identifier
     - Current status (baseline, improved, etc.)
     - Brief description
   - User can select an agent to view details

2. **Agent Detail View**
   - Shows the selected agent's current prompt
   - Displays available tools/MCPs
   - Shows conversation history and evaluation results
   - **Key Feature**: Chat interface in Cursor-style format
     - User can type: `@agent1 is doing X. It should do Y.`
     - This sets the SOP criteria for evaluation

### Demo Scenario: Weather Agent

**SOP Goal**: "Provide accurate information about the user's weather"

**Step 1: Baseline Agent Failure**
- Display a conversation where the baseline agent fails to meet the SOP
- Show example: User asks "What's the weather?" → Agent responds generically without actually fetching weather data
- Highlight this as a "wrongful completion" - the agent doesn't have access to weather data

**Step 2: System Activation**
- When user sets SOP criteria, the system automatically:
  1. Generates multiple conversation scenarios (Completions Layer)
  2. Evaluates each conversation against SOP criteria (Judging Layer)
  3. Identifies failures (SOP_met = false)
  4. Spawns parallel fixer agents (Simulation Layer)

**Step 3: Fixer Agents at Work**
- Visualize fixer agents analyzing the failure
- Show them receiving:
  - The agent file (current prompt + tools)
  - The failed SOP criterion ("provide accurate weather information")
  - The violating conversation
  - Available MCP servers (including weather APIs, Perplexity, etc.)
- Display fixer agents suggesting changes:
  - Adding a weather MCP server
  - Modifying the prompt to include instructions for using the weather tool
  - Adding location-gathering capabilities

**Step 4: Merge Process**
- Show the merge layer combining multiple fixes
- Display the diff/changes being applied to the agent file
- Highlight if LLM-assisted merging is needed for conflicts

**Step 5: Re-evaluation**
- System automatically re-runs conversations with the improved agent
- Show a successful conversation example:
  - User: "What's the weather?"
  - Agent: "What's your location?" (asks for location)
  - User: "San Francisco"
  - Agent: [Calls weather MCP] "The weather in San Francisco is..."
- Display updated evaluation results showing SOP_met = true

**Step 6: Success State**
- Show the improved agent file with new tools/MCPs
- Display success metrics:
  - Pass rate improvement
  - Number of iterations
  - Changes made
- **"Deploy Agent" button**: Shows a popup/modal indicating the agent will be deployed (demo-only, not functional)

## Architecture Integration

The frontend must visualize and interact with the backend architecture:

### Backend System (Already Implemented)
1. **Completions Layer**: Generates multiple conversations between base agent and conversational agents
2. **Judging Layer**: Evaluates conversations against SOP criteria using judge LLM
3. **Simulation Layer**: Spawns parallel fixer agents for each failed criterion
4. **Merge Layer**: Intelligently merges multiple modifications
5. **Orchestration Layer**: Coordinates the full improvement loop

### Frontend Visualization Requirements

**Real-time Progress Display**:
- Show conversation generation progress (multiple parallel conversations)
- Display evaluation results as they come in (boolean checklist)
- Visualize fixer agents working in parallel (one per failed criterion)
- Show merge process and conflict resolution
- Display iteration count and feedback loop status

**Conversation Viewer**:
- Display conversations in a chat-like interface
- Highlight violations (where SOP criteria failed)
- Show successful conversations after fixes
- Allow users to expand/collapse conversation details

**Agent File Editor**:
- Show the agent's Python file (Dedalus agent format)
- Display diffs/changes as they're applied
- Highlight new tools/MCPs added
- Show prompt modifications

**Evaluation Dashboard**:
- SOP criteria checklist with boolean indicators
- Pass rate visualization (charts/graphs)
- Iteration history timeline
- Success/failure metrics

## Key UI Components Needed

1. **Agent Card Grid**: Display available agents
2. **Chat Interface**: Cursor-style input for setting SOP criteria (`@agent1 is doing X. It should do Y`)
3. **Conversation Viewer**: Chat-style display of agent conversations
4. **Progress Indicators**: Show system processing (conversations, judging, fixing, merging)
5. **Evaluation Panel**: SOP checklist with pass/fail indicators
6. **Code Diff Viewer**: Show changes to agent files
7. **Metrics Dashboard**: Charts showing improvement over iterations
8. **Deploy Modal**: Popup for "Deploy Agent" action

## User Journey

1. **Landing**: User sees available agents
2. **Selection**: User selects an agent to improve
3. **SOP Definition**: User types in chat: `@weather-agent is giving generic weather responses. It should provide accurate, location-specific weather information.`
4. **System Processing**: 
   - User watches as system generates conversations
   - Sees evaluation results (some failures)
   - Observes fixer agents analyzing and suggesting changes
   - Watches merge process combine fixes
   - Sees re-evaluation with improved results
5. **Success**: User sees improved agent with new weather MCP, successful conversations, and "Deploy Agent" option

## Technical Integration Points

### API Endpoints Needed
- `GET /api/agents` - List available agents
- `GET /api/agents/:id` - Get agent details
- `POST /api/agents/:id/improve` - Trigger improvement process
  - Body: `{ sop_criteria: string[], initial_message: string }`
- `GET /api/agents/:id/status` - Get improvement status/progress
- `GET /api/agents/:id/conversations` - Get conversation history
- `GET /api/agents/:id/evaluations` - Get evaluation results
- `GET /api/agents/:id/file` - Get current agent file

### WebSocket/SSE for Real-time Updates
- Stream conversation generation progress
- Stream evaluation results
- Stream fixer agent suggestions
- Stream merge process updates
- Stream re-evaluation results

## Design Principles

1. **Clarity**: Make the complex orchestration process easy to understand
2. **Visual Feedback**: Show progress at every step
3. **Transparency**: Display what's happening under the hood
4. **Interactivity**: Allow users to explore conversations and changes
5. **Modern Aesthetics**: Clean, professional YC startup style

## Success Criteria

The demo successfully demonstrates:
- ✅ Agent failing to meet SOP criteria
- ✅ Automatic detection of failures
- ✅ Parallel fixer agents suggesting improvements
- ✅ Intelligent merging of multiple fixes
- ✅ Successful re-evaluation showing improvement
- ✅ Visual representation of the entire orchestration flow
- ✅ User can understand how agents are improved automatically

## Implementation Notes

- Backend system is fully implemented and tested
- Frontend should be built as a separate Next.js application
- Use shadcn-ui for consistent, modern components
- Implement real-time updates via WebSocket or Server-Sent Events
- Ensure responsive design (works on desktop and tablet)
- Use code syntax highlighting for agent files and diffs
- Implement smooth animations for state transitions

## Example User Interactions

**Setting SOP Criteria**:
```
User types: @weather-agent is giving generic responses. It should provide accurate, location-specific weather data by calling weather APIs.
```

**Viewing Results**:
- User can click on any conversation to see full details
- User can expand evaluation results to see reasoning
- User can view the diff of agent file changes
- User can see which MCPs/tools were added

**Deploying**:
- User clicks "Deploy Agent" button
- Modal appears: "Agent will be deployed and changes will be reflected. [OK] [Cancel]"
- This is demo-only, no actual deployment happens

---

**This specification should be used as a prompt for AI agents to build the frontend demo platform. The backend orchestration system is already complete and ready for integration.**
