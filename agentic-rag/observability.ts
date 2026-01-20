import 'dotenv/config';
import { v4 as uuidv4 } from 'uuid';

const LOGS_BASE = 'https://api.staging.adaline.ai/v2/logs';
const apiKey = process.env.ADALINE_API_KEY;
if (!apiKey) throw new Error('ADALINE_API_KEY missing');

export interface Span {
  name: string;
  status: 'success' | 'error';
  startedAt: number;
  endedAt: number;
  content: any;
  referenceId?: string;
  parentReferenceId?: string;
  promptId?: string;
  deploymentId?: string;
  sessionId?: string;
  runEvaluation?: boolean;
  attributes?: Record<string, string | number | boolean>;
  tags?: string[];
  cost?: number;
  tokens?: { input?: number; output?: number; total?: number };
}

export interface Trace {
  name: string;
  status: 'success' | 'error';
  startedAt: number;
  endedAt: number;
  referenceId: string;
  spans: Span[];
  projectId: string;
  promptId?: string;
  deploymentId?: string;
  sessionId?: string;
  attributes?: Record<string, string | number | boolean>;
  tags?: string[];
}

export function createTrace(name: string, projectId: string, promptId?: string, deploymentId?: string): Trace {
  return {
    name,
    status: 'success',
    startedAt: Date.now(),
    endedAt: 0,
    referenceId: uuidv4(),
    spans: [],
    projectId,
    promptId,
    deploymentId,
    sessionId: uuidv4(),
    attributes: {
      app_name: 'Agentic-RAG',
      runtime: 'node',
    },
    tags: ['agentic-rag'],
  };
}

export function addSpan(trace: Trace, span: Omit<Span, 'sessionId'>): Trace {
  const newSpan: Span = {
    ...span,
    sessionId: trace.sessionId,
    attributes: {
      ...(span.attributes || {}),
    },
    tags: Array.from(new Set([...(span.tags || []), span.name])),
  };
  trace.spans.push(newSpan);
  return trace;
}

export async function submitTrace(trace: Trace): Promise<void> {
  if (!apiKey) {
    console.warn("ADALINE_API_KEY is not set. Skipping trace submission.");
    return;
  }

  trace.endedAt = Date.now();
  if (trace.spans.some(s => s.status === 'error')) {
    trace.status = 'error';
  }

  const mapStatus = (s: 'success' | 'error'): 'success' | 'failure' => (s === 'success' ? 'success' : 'failure');

  const traceEnded = trace.endedAt && trace.endedAt > trace.startedAt ? trace.endedAt : trace.startedAt + 1;
  const tracePayload = {
    startedAt: trace.startedAt,
    endedAt: traceEnded,
    name: trace.name,
    status: mapStatus(trace.status),
    referenceId: trace.referenceId,
    sessionId: trace.sessionId,
    attributes: trace.attributes,
    tags: trace.tags,
  };

  const spansPayload = trace.spans.map((s) => {
    const c = s.content ?? {};
    const type = c.type as string | undefined;

    let content: any = {};
    if (type) {
      content.type = type;
    }
    if (c.input !== undefined) {
      content.input = typeof c.input === 'string' ? c.input : JSON.stringify(c.input);
    }
    
    if (c.output !== undefined && type !== 'Model') {
      const outputObj = typeof c.output === 'object' ? c.output : { value: c.output };
      const enrichedOutput = {
        ...outputObj,
        ...(s.cost !== undefined && { cost: s.cost }),
        ...(s.tokens && { tokens: s.tokens }),
      };
      content.output = JSON.stringify(enrichedOutput);
    } else if (c.output === undefined && type !== 'Model' && (s.cost !== undefined || s.tokens)) {
      content.output = JSON.stringify({
        ...(s.cost !== undefined && { cost: s.cost }),
        ...(s.tokens && { tokens: s.tokens }),
      });
    }
    
    if (type === 'Model') {
      // For Model type, provider/model are top-level in content
      content.provider = c.provider ?? 'openai';
      content.model = c.model ?? '';
      
      // Add cost for Model type (per API spec - only supported for Model)
      if (s.cost !== undefined && s.cost !== null) {
        content.cost = s.cost;
        console.log(`[DEBUG] Adding cost to Model span "${s.name}": ${s.cost}`);
      } else {
        console.log(`[DEBUG] No cost for Model span "${s.name}": cost=${s.cost}`);
      }
      
      if (c.output !== undefined) {
        const outputObj = typeof c.output === 'object' ? c.output : JSON.parse(JSON.stringify(c.output));
        const parsed = typeof outputObj === 'string' ? JSON.parse(outputObj || '{}') : outputObj;
        
        if (s.tokens) {
          parsed.tokenUsage = {
            promptTokens: s.tokens.input || 0,
            completionTokens: s.tokens.output || 0,
            totalTokens: s.tokens.total || 0,
          };
        }
        
        content.output = JSON.stringify(parsed);
      }
    }

    const ended = s.endedAt && s.endedAt > s.startedAt ? s.endedAt : s.startedAt + 1;
    
    const spanPayload: any = {
      startedAt: s.startedAt,
      endedAt: ended,
      name: s.name,
      status: mapStatus(s.status),
      content,
      referenceId: s.referenceId,
      parentReferenceId: s.parentReferenceId,
      promptId: s.promptId,
      deploymentId: s.deploymentId,
      sessionId: s.sessionId,
      attributes: s.attributes,
      tags: s.tags,
    };
    
    // For Model spans, add variables at content level (embedded in content object before submission)
    if (type === 'Model' && c.variables) {
      // Variables go in content but we add them here after content is built
      spanPayload.content.variables = c.variables;
    }
    
    if (s.runEvaluation !== undefined) {
      spanPayload.runEvaluation = s.runEvaluation;
    }
    
    return spanPayload;
  });

  // Retry logic for network errors
  const maxRetries = 3;
  const retryDelay = 1000; // Start with 1 second
  let lastError: any = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const payload = { projectId: trace.projectId, trace: tracePayload, spans: spansPayload };
      
      // Debug: log final_response span structure (only on first attempt)
      if (attempt === 0) {
        const finalResponseSpan = spansPayload.find(s => s.name === 'final_response');
        if (finalResponseSpan) {
          console.log('\n=== DEBUG: final_response span structure ===');
          console.log(JSON.stringify(finalResponseSpan, null, 2));
        }
      }
      
      // Create AbortController for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
      
      const response = await fetch(`${LOGS_BASE}/trace`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`,
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorBody = await response.text();
        const errorMsg = `Failed to submit trace. Status: ${response.status}. Body: ${errorBody}`;
        if (attempt < maxRetries) {
          console.warn(`${errorMsg} (Attempt ${attempt + 1}/${maxRetries + 1}, retrying...)`);
          await new Promise(resolve => setTimeout(resolve, retryDelay * Math.pow(2, attempt)));
          continue;
        } else {
          console.error(errorMsg);
          return;
        }
      } else {
        const result: any = await response.json();
        console.log(`✓ Trace submitted successfully (ID: ${result.traceId || trace.referenceId})`);
        return; // Success, exit retry loop
      }
    } catch (error: any) {
      lastError = error;
      const isNetworkError = error?.code === 'ECONNRESET' || 
                            error?.code === 'ECONNREFUSED' || 
                            error?.code === 'ENOTFOUND' ||
                            error?.name === 'AbortError' ||
                            error?.message?.includes('fetch failed');
      
      if (isNetworkError && attempt < maxRetries) {
        const delay = retryDelay * Math.pow(2, attempt);
        console.warn(`Network error submitting trace (${error?.code || error?.name || 'unknown'}): ${error?.message || error}. Retrying in ${delay}ms... (Attempt ${attempt + 1}/${maxRetries + 1})`);
        await new Promise(resolve => setTimeout(resolve, delay));
        continue;
      } else {
        // Last attempt failed or non-network error
        if (isNetworkError) {
          console.warn(`Failed to submit trace after ${maxRetries + 1} attempts due to network error: ${error?.code || error?.name || 'unknown'}. This is non-critical - execution completed successfully.`);
        } else {
          console.error("Error submitting trace:", error);
        }
        return; // Exit retry loop
      }
    }
  }
}

export function now() { return Date.now(); }
