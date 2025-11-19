import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';

const execAsync = promisify(exec);

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { criteria } = body;

    if (!criteria || !Array.isArray(criteria)) {
      return NextResponse.json(
        { error: 'Invalid criteria provided' },
        { status: 400 }
      );
    }

    // Path to Python script (in project root)
    const scriptPath = path.join(process.cwd(), '..', 'run_orchestration.py');
    const workingDir = path.join(process.cwd(), '..');
    // Use virtual environment Python
    const pythonPath = path.join(workingDir, '.venv', 'bin', 'python3');
    
    // Create input JSON
    const inputJson = JSON.stringify({ criteria });
    
    // Execute Python script with stdin using venv Python
    const child = require('child_process').spawn(pythonPath, [scriptPath], {
      cwd: workingDir,
      stdio: ['pipe', 'pipe', 'pipe']
    });
    
    // Write input to stdin
    child.stdin.write(inputJson);
    child.stdin.end();
    
    // Collect output
    let stdout = '';
    let stderr = '';
    
    child.stdout.on('data', (data: Buffer) => {
      stdout += data.toString();
    });
    
    child.stderr.on('data', (data: Buffer) => {
      stderr += data.toString();
    });
    
    // Wait for process to complete
    await new Promise((resolve, reject) => {
      child.on('close', (code: number) => {
        if (code !== 0) {
          reject(new Error(`Python script exited with code ${code}: ${stderr}`));
        } else {
          resolve(null);
        }
      });
      child.on('error', reject);
    });

    if (stderr) {
      console.error('Python stderr:', stderr);
    }

    // Parse JSON output from Python script
    const result = JSON.parse(stdout);
    
    return NextResponse.json(result);
    
  } catch (error: any) {
    console.error('Orchestration error:', error);
    return NextResponse.json(
      { error: error.message || 'Orchestration failed' },
      { status: 500 }
    );
  }
}