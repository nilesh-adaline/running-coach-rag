/**
 * Simple UI server for the multi-agent RAG orchestrator.
 * - Serves static files from ./public
 * - POST /api/run: body { CONTEXT, RUN_BLOCK, WHAT_TO_COVER, NOTES } -> { finalResponse } or { error }
 */
import path from 'path';
import { config } from 'dotenv';
import { existsSync } from 'fs';

// Load .env from project root so RAG (Pinecone, OAI_API_KEY) works when server runs from any dir
const cwdEnv = path.join(process.cwd(), '.env');
const parentEnv = path.join(process.cwd(), '..', '.env');
config({ path: existsSync(cwdEnv) ? cwdEnv : parentEnv });
import express from 'express';
import cors from 'cors';
import { orchestrateMultiAgentWithHandoffs, getOrCreateTrace, safeSubmitTrace, clearTraceForNextRequest } from './orchestrator';

const app = express();
const PORT = process.env.UI_PORT ?? 3000;

app.use(cors());
app.use(express.json({ limit: '1mb' }));

// Static files (UI) — resolve correctly whether run from project root or agentic-rag/
const cwd = process.cwd();
const publicDir = (() => {
  const a = path.join(cwd, 'agentic-rag', 'public');
  const b = path.join(cwd, 'public');
  return existsSync(a) ? a : (existsSync(b) ? b : a);
})();
app.use(express.static(publicDir));

app.post('/api/run', async (req, res) => {
  const { CONTEXT, RUN_BLOCK, WHAT_TO_COVER, NOTES } = req.body ?? {};
  if (!CONTEXT || !RUN_BLOCK || !WHAT_TO_COVER || !NOTES) {
    return res.status(400).json({
      error: 'Missing required fields: CONTEXT, RUN_BLOCK, WHAT_TO_COVER, NOTES',
    });
  }

  const userInputs = {
    CONTEXT: String(CONTEXT).trim(),
    RUN_BLOCK: String(RUN_BLOCK).trim(),
    WHAT_TO_COVER: String(WHAT_TO_COVER).trim(),
    NOTES: String(NOTES).trim(),
  };

  try {
    // Don't pass a parent ref so the root span has no parentReferenceId (no CLI root span when running from UI).
    const result = await orchestrateMultiAgentWithHandoffs(userInputs);
    const trace = getOrCreateTrace();
    await safeSubmitTrace(trace);
    clearTraceForNextRequest();
    res.json({ finalResponse: result.finalResponse });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error('Orchestrator error:', message);
    clearTraceForNextRequest();
    res.status(500).json({ error: message });
  }
});

// Fallback: serve index.html for SPA-style routing (optional)
app.get('*', (_req, res) => {
  res.sendFile(path.join(publicDir, 'index.html'));
});

const MAX_PORT_ATTEMPTS = 5;
function tryListen(port: number, attempt: number): void {
  const server = app.listen(port, () => {
    console.log(`\n🌐 UI server: http://localhost:${port}`);
    console.log(`   POST /api/run with { CONTEXT, RUN_BLOCK, WHAT_TO_COVER, NOTES }\n`);
  });
  server.on('error', (err: NodeJS.ErrnoException) => {
    if (err.code === 'EADDRINUSE' && attempt < MAX_PORT_ATTEMPTS) {
      server.close();
      console.warn(`Port ${port} in use, trying ${port + 1}...`);
      tryListen(port + 1, attempt + 1);
    } else {
      console.error(err.message || err);
      process.exit(1);
    }
  });
}
tryListen(PORT, 0);
