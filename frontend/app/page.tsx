"use client";

import { useState } from "react";

function IterationDetails({ iteration }: { iteration: any }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showConversations, setShowConversations] = useState(false);
  const [showPromptChanges, setShowPromptChanges] = useState(false);

  return (
    <div className="rounded-lg border bg-white shadow-sm">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-6 text-left hover:bg-gray-50"
      >
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-gray-900">
            Iteration {iteration.iteration}
            {iteration.all_criteria_passed && (
              <span className="ml-2 rounded-full bg-green-100 px-2 py-1 text-xs text-green-700">
                All Passed
              </span>
            )}
            {!iteration.all_criteria_passed && (
              <span className="ml-2 rounded-full bg-red-100 px-2 py-1 text-xs text-red-700">
                {iteration.total_passed}/{iteration.total_conversations} Passed
              </span>
            )}
          </h3>
          <span className="text-gray-400">{isExpanded ? "▼" : "▶"}</span>
        </div>
      </button>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="border-t p-6 space-y-4">
          
          {/* Actual Conversations */}
          <div className="rounded bg-purple-50 p-4">
            <button
              onClick={() => setShowConversations(!showConversations)}
              className="w-full text-left"
            >
              <h4 className="font-medium text-gray-900 flex items-center justify-between">
                <span>View Conversations ({iteration.total_conversations} total)</span>
                <span className="text-gray-400">{showConversations ? "▼" : "▶"}</span>
              </h4>
            </button>
            {showConversations && iteration.judgment_result?.judgments && (
              <div className="mt-3 space-y-3">
                {iteration.judgment_result.judgments.map((judgment: any, jIdx: number) => {
                  // Get the actual conversation from the iteration data
                  const conversations = iteration.judgment_result?.conversations || [];
                  const conversation = conversations[jIdx];
                  
                  return (
                    <div key={jIdx} className="rounded bg-white p-3 border border-purple-200">
                      <p className="text-xs font-medium text-purple-900 mb-2">
                        Conversation {jIdx + 1} - {judgment.overall_pass ? "✓ Passed" : "✗ Failed"}
                      </p>
                      {conversation?.messages && (
                        <div className="space-y-2">
                          {conversation.messages.map((msg: any, mIdx: number) => (
                            <div
                              key={mIdx}
                              className={`text-xs p-2 rounded ${
                                msg.role === "base_agent"
                                  ? "bg-blue-50 border-l-2 border-blue-500"
                                  : "bg-gray-50 border-l-2 border-gray-500"
                              }`}
                            >
                              <span className="font-medium">
                                {msg.role === "base_agent" ? "Agent:" : "User:"}
                              </span>
                              <p className="mt-1 whitespace-pre-wrap">{msg.content}</p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Judgment Results */}
          {iteration.judgment_result?.judgments && (
            <div className="rounded bg-gray-50 p-4">
              <h4 className="font-medium text-gray-900 mb-3">Judgment Results</h4>
              {iteration.judgment_result.judgments.map((judgment: any, jIdx: number) => (
                <div key={jIdx} className="mb-4 last:mb-0">
                  <p className="text-sm font-medium text-gray-700 mb-2">
                    Conversation {jIdx + 1}: {judgment.overall_pass ? "✓ Passed" : "✗ Failed"}
                  </p>
                  <div className="space-y-1">
                    {judgment.criteria_scores?.map((score: any, sIdx: number) => (
                      <div key={sIdx} className="flex items-start text-xs">
                        <span className={score.met ? "text-green-600" : "text-red-600"}>
                          {score.met ? "✓" : "✗"}
                        </span>
                        <span className="ml-2 flex-1">
                          <span className="font-medium">{score.criterion}</span>
                          {score.reasoning && (
                            <span className="block text-gray-600 mt-1">{score.reasoning}</span>
                          )}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Modification Applied */}
          {iteration.modification_applied && (
            <div className="rounded bg-blue-50 p-4">
              <h4 className="font-medium text-gray-900 mb-2">Fixer Agent Suggestion</h4>
              <p className="text-sm text-gray-700 mb-2">
                <span className="font-medium">Failed criterion:</span> {iteration.modification_applied.criterion}
              </p>
              
              {/* Show prompt changes */}
              <button
                onClick={() => setShowPromptChanges(!showPromptChanges)}
                className="text-sm text-blue-600 hover:text-blue-800 mb-2"
              >
                {showPromptChanges ? "Hide" : "Show"} Prompt Changes
              </button>
              
              {showPromptChanges && (
                <div className="mt-2 mb-3 rounded bg-white p-3 border border-blue-200">
                  <p className="text-xs font-medium text-gray-700 mb-2">Original Prompt:</p>
                  <pre className="text-xs bg-gray-50 p-2 rounded mb-3 whitespace-pre-wrap">
                    {iteration.modification_applied.original_prompt}
                  </pre>
                  <p className="text-xs font-medium text-gray-700 mb-2">Modified Prompt:</p>
                  <pre className="text-xs bg-green-50 p-2 rounded whitespace-pre-wrap">
                    {iteration.modification_applied.modified_prompt}
                  </pre>
                </div>
              )}
              
              {iteration.modification_applied.mcp_servers_added && iteration.modification_applied.mcp_servers_added.length > 0 && (
                <div className="mb-2">
                  <p className="text-sm font-medium text-gray-700">MCP Servers Added:</p>
                  <ul className="list-disc list-inside text-xs text-gray-600">
                    {iteration.modification_applied.mcp_servers_added.map((server: string, i: number) => (
                      <li key={i} className="font-mono">{server}</li>
                    ))}
                  </ul>
                </div>
              )}
              {iteration.modification_applied.changes_made && (
                <div>
                  <p className="text-sm font-medium text-gray-700 mb-1">Changes Made:</p>
                  <ul className="space-y-1">
                    {iteration.modification_applied.changes_made.map((change: string, i: number) => (
                      <li key={i} className="text-xs text-gray-600">• {change}</li>
                    ))}
                  </ul>
                </div>
              )}
              {iteration.modification_applied.reasoning && (
                <div className="mt-2">
                  <p className="text-xs font-medium text-gray-700">Fixer's Reasoning:</p>
                  <p className="text-xs text-gray-600 italic mt-1">
                    {iteration.modification_applied.reasoning}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function Home() {
  const [criteria, setCriteria] = useState("Provide accurate information about the user's weather");
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [showDeployPopup, setShowDeployPopup] = useState(false);

  const handleRunImprovement = async () => {
    setIsRunning(true);
    setResult(null);

    try {
      console.log("Starting orchestration request...");
      
      // Call Next.js API route that executes Python script
      const response = await fetch("/api/orchestrate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          criteria: criteria.split("\n").filter((c) => c.trim()),
        }),
      });

      console.log("Response received, parsing JSON...");
      const data = await response.json();
      console.log("Orchestration complete:", data);
      setResult(data);
    } catch (error) {
      console.error("Orchestration error:", error);
      setResult({ error: "Failed to run orchestration" });
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <div className="border-b bg-white">
        <div className="mx-auto max-w-6xl px-6 py-4">
          <h1 className="text-2xl font-semibold text-gray-900">Agent Orchestration Platform</h1>
          <p className="text-sm text-gray-600">Self-improving AI agents for enterprise</p>
        </div>
      </div>

      <div className="mx-auto max-w-6xl px-6 py-8">
        {/* Input Section */}
        <div className="mb-8 rounded-lg border bg-white p-6 shadow-sm">
          <h2 className="mb-4 text-lg font-semibold text-gray-900">Standard Operating Procedures (SOP)</h2>
          <p className="mb-4 text-sm text-gray-600">
            Define the criteria your agent must meet. One criterion per line.
          </p>
          <textarea
            value={criteria}
            onChange={(e) => setCriteria(e.target.value)}
            className="w-full rounded-md border border-gray-300 p-3 font-mono text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
            rows={3}
            placeholder="Enter SOP criteria (one per line)"
          />
          <button
            onClick={handleRunImprovement}
            disabled={isRunning}
            className="mt-4 rounded-md bg-blue-600 px-6 py-2 font-medium text-white hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            {isRunning ? "Running orchestration..." : "Run Improvement"}
          </button>
          {isRunning && (
            <div className="mt-3 text-sm text-gray-600">
              <div className="flex items-center space-x-2">
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-gray-300 border-t-blue-600"></div>
                <span>Processing agent improvement (this may take 30-60 seconds)...</span>
              </div>
              <p className="mt-2 text-xs text-gray-500">
                Running conversations → Judging criteria → Simulating fixes → Merging improvements
              </p>
            </div>
          )}
        </div>

        {/* Results Section */}
        {result && (
          <div className="space-y-6">
            {/* Summary */}
            <div className="rounded-lg border bg-white p-6 shadow-sm">
              <h2 className="mb-4 text-lg font-semibold text-gray-900">Orchestration Results</h2>
              {result.error ? (
                <p className="text-red-600">{result.error}</p>
              ) : (
                <div className="space-y-2">
                  <p className="text-sm">
                    <span className="font-medium">Status:</span>{" "}
                    <span className={result.success ? "text-green-600" : "text-yellow-600"}>
                      {result.status}
                    </span>
                  </p>
                  <p className="text-sm">
                    <span className="font-medium">Total Iterations:</span> {result.total_iterations}
                  </p>
                  <p className="text-sm">
                    <span className="font-medium">All Criteria Passed:</span>{" "}
                    <span className={result.all_criteria_passed ? "text-green-600" : "text-red-600"}>
                      {result.all_criteria_passed ? "Yes" : "No"}
                    </span>
                  </p>
                </div>
              )}
            </div>

            {/* Iterations */}
            {result.iterations && (
              <div className="space-y-4">
                {result.iterations.map((iteration: any, idx: number) => (
                  <IterationDetails key={idx} iteration={iteration} />
                ))}
              </div>
            )}

            {/* Deploy Button */}
            {result.success && (
              <div className="rounded-lg border bg-white p-6 shadow-sm">
                <button
                  onClick={() => setShowDeployPopup(true)}
                  className="rounded-md bg-green-600 px-6 py-2 font-medium text-white hover:bg-green-700"
                >
                  Deploy Agent
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Deploy Popup */}
      {showDeployPopup && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
          <div className="rounded-lg bg-white p-6 shadow-xl">
            <h3 className="mb-4 text-lg font-semibold text-gray-900">Deploy Agent</h3>
            <p className="mb-6 text-sm text-gray-600">
              Agent improvements will be relayed and reflected in your production environment.
            </p>
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => setShowDeployPopup(false)}
                className="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}