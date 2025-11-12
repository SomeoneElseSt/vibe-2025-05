# Agent Orchestration System

A backend system that orchestrates AI agents to improve their prompts through iterative conversation generation, evaluation, and automated fixes.

## Architecture

```mermaid
flowchart TD
    Start([Start: Agent Baseline]) --> CompletionsLayer[Completions Layer]
    
    subgraph CompletionsLayer[Completions Layer]
        Agent[Agent Baseline]
        SOP[From SOPs / User Request]
        Agent --> CO1[CO]
        Agent --> CO2[CO]
        Agent --> CO3[CO]
        Agent --> CO4[CO]
        Agent --> CO5[CO]
        SOP --> CO1
        SOP --> CO2
        SOP --> CO3
        SOP --> CO4
        SOP --> CO5
        CO1 --> Conversations[C Conversations]
        CO2 --> Conversations
        CO3 --> Conversations
        CO4 --> Conversations
        CO5 --> Conversations
    end
    
    Conversations --> JudgingLayer[Judging Layer]
    
    subgraph JudgingLayer[Judging Layer]
        JudgeLLM[C - Judge LLM]
        SOPGoals[SOP Goals → Checklist]
        Checklist[bool, bool, etc...]
        JudgeLLM --> SOPGoals
        SOPGoals --> Checklist
        Checklist --> Decision{All bools true?}
        Decision -->|Yes| Success[Confirm: Agent Works!]
        Decision -->|No| WrongfulCompletion[Save Wrongful Completion]
    end
    
    WrongfulCompletion --> SimulationLayer[Simulation Layer]
    
    subgraph SimulationLayer[Simulation Layer]
        ForEach[For each false_bool:]
        FixerAgent[Spin up Fixer Agent Instance]
        FixerInput[Given:
        - Agent file
        - Desired goal
        - Completion violating goal
        - Dedalus MCP list
        - Instructions for adding tools]
        ForEach --> FixerAgent
        FixerAgent --> FixerInput
        FixerInput --> AgentOutput[Agent outputs changed file/diff:
        - diff prompt
        - new tools / MCPs]
    end
    
    AgentOutput --> MergeLayer[Merge Layer]
    
    subgraph MergeLayer[Merge Layer]
        SingleFix[Single Fixed Dedalus Agent File<br/>w/ summary of changes]
        MergeLogic{Multiple fixes?}
        SingleFix --> MergeLogic
        AgentOutput --> MergeLogic
        MergeLogic -->|Yes| TryMerge[Try to merge individual files]
        TryMerge --> MergeCheck{Merge issues?}
        MergeCheck -->|Yes| LLMMerge[Call LLM to output merged file]
        MergeCheck -->|No| MergedFile[Merged File]
        LLMMerge --> MergedFile
        MergeLogic -->|No| MergedFile
    end
    
    MergedFile --> FeedbackLoop{For each new_file:}
    FeedbackLoop -->|Re-run| CompletionsLayer
    FeedbackLoop -->|Re-judge| JudgingLayer
    FeedbackLoop -->|Continue| SimulationLayer
    FeedbackLoop -->|Conclude| End([End: Improved Agent])
    
    Success --> End
    
    style CompletionsLayer fill:#e1f5e1
    style JudgingLayer fill:#e1f5e1
    style SimulationLayer fill:#e1f5e1
    style MergeLayer fill:#e1f5e1
    style Success fill:#90ee90
    style End fill:#90ee90
```

## System Layers

### 1. Completions Layer
Generates multiple conversations between a base agent and various conversational agents with different prompts. Creates parallel conversation instances to test agent behavior across different scenarios.

**Input**: Base agent prompt, conversational agent prompts, initial message  
**Output**: List of conversations

### 2. Judging Layer
Evaluates conversations against SOP (Standard Operating Procedure) goals using a judge LLM. Each criterion is evaluated as a boolean, creating a checklist of requirements.

**Input**: Conversations, SOP goals/criteria, judge prompt  
**Output**: Boolean checklist for each criterion

### 3. Simulation Layer
For each failed criterion, spawns a parallel "fixer" agent that analyzes the failure and suggests modifications. Each fixer agent receives:
- The agent file
- The failed goal/criterion
- The violating conversation
- Available Dedalus MCP servers list
- Instructions for adding tools

**Input**: Failed criteria, conversations, agent file  
**Output**: Modified agent files/diffs with prompt changes, new tools, or new MCP servers

### 4. Merge Layer
Intelligently merges multiple modifications from fixer agents into a single coherent agent file. Uses automatic merging for simple cases and LLM-assisted merging for complex conflicts.

**Input**: Multiple modified agent files/diffs  
**Output**: Single merged agent file

### 5. Feedback Loop
The merged file triggers a new iteration:
- Re-run conversations layer
- Re-judge with judging layer
- Continue to simulation layer if needed
- Conclude when all criteria pass

## Tech Stack

- **Python** - Core backend logic
- **UV** - Package management
- **Dedalus Agents SDK** - Agent framework
- **Pydantic** - Type validation and structured outputs
- **Async/Await** - Parallel execution throughout

## Project Status

✅ **All core components complete and tested**
- Conversations Layer (async, parallel)
- Judging Layer (async, parallel, structured outputs)
- Simulation Layer (async, parallel, enhanced MCP support)
- Merge Layer (automatic + LLM-assisted)
- Orchestration Layer (full improvement loop)
- File Manager (read/write agent files)

See `AGENTS.md` for detailed implementation documentation.
