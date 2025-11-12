"use client";

import { useState } from "react";

export default function Home() {
  const [criteria, setCriteria] = useState("Provide accurate information about the user's weather");
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [showDeployPopup, setShowDeployPopup] = useState(false);

  const handleRunImprovement = async () => {
    setIsRunning(true);
    setResult(null);

    try {
      const response = await fetch("http://localhost:8000/api/orchestrate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          criteria: criteria.split("\n").filter((c) => c.trim()),
        }),
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error:", error);
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
            className="mt-4 rounded-md bg-blue-600 px-6 py-2 font-medium text-white hover:bg-blue-700 disabled:bg-gray-400"
          >
            {isRunning ? "Running..." : "Run Improvement"}
          </button>
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
                  <div key={idx} className="rounded-lg border bg-white p-6 shadow-sm">
                    <h3 className="mb-3 font-semibold text-gray-900">
                      Iteration {iteration.iteration}
                      {iteration.all_criteria_passed && (
                        <span className="ml-2 rounded-full bg-green-100 px-2 py-1 text-xs text-green-700">
                          All Passed
                        </span>
                      )}
                    </h3>
                    <div className="space-y-2 text-sm">
                      <p>
                        <span className="font-medium">Conversations:</span> {iteration.total_conversations} |{" "}
                        <span className="font-medium">Passed:</span> {iteration.total_passed}
                      </p>
                      {iteration.modification_applied && (
                        <div className="mt-3 rounded bg-gray-50 p-3">
                          <p className="font-medium text-gray-700">Modification Applied:</p>
                          <p className="mt-1 text-xs text-gray-600">
                            Criterion: {iteration.modification_applied.criterion}
                          </p>
                          {iteration.modification_applied.changes_made && (
                            <ul className="mt-2 list-inside list-disc space-y-1 text-xs text-gray-600">
                              {iteration.modification_applied.changes_made.map((change: string, i: number) => (
                                <li key={i}>{change}</li>
                              ))}
                            </ul>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
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