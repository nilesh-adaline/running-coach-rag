import { v4 as uuidv4 } from 'uuid';
import { PROJECT_ID as FETCHED_PROJECT_ID } from './fetchDeployedPrompt';

const ADALINE_API_KEY = process.env.ADALINE_API_KEY;
const LOGS_BASE_URL = "https://api.staging.adaline.ai/v2/logs";


export interface Span {
  name: string;
  status: 'success' | 'error';
  startedAt: number; // Unix timestamp in milliseconds
  endedAt: number; // Unix timestamp in milliseconds
  content: any; // This will be one of the content types below
  traceId?: string;
  traceReferenceId?: string;
  // Optional additional fields
  referenceId?: string;
  parentReferenceId?: string;
  promptId?: string;
  deploymentId?: string;
  sessionId?: string;
  runEvaluation?: boolean; // Whether to run evaluation on this span and enable "Add to Dataset"
  attributes?: Record<string, string | number | boolean>;
  tags?: string[];
  // Metrics stored internally but sent via content
  cost?: number; // Cost in USD
  latency?: number; // Latency in milliseconds
  tokens?: { input?: number; output?: number; total?: number }; // Token usage for LLM calls
}

export interface Trace {
  name: string;
  status: 'success' | 'error';
  startedAt: number; // Unix timestamp in milliseconds
  endedAt: number; // Unix timestamp in milliseconds
  referenceId: string;
  spans: Span[];
  projectId?: string;
  sessionId?: string;
  attributes?: Record<string, string | number | boolean>;
  tags?: string[];
}

// Main function to create a trace
export function createTrace(name: string, projectId?: string): Trace {
  return {
    name,
    status: 'success', // Default status
    startedAt: Date.now(),
    endedAt: 0, // Will be set when the trace is finalized
    referenceId: uuidv4(),
    spans: [],
    projectId,
    sessionId: uuidv4(),
    attributes: {
      app_name: 'The Running Coach App (RAG)',
      runtime: 'node',
      language: 'ts',
      nodeEnv: process.env.NODE_ENV || 'development',
    },
    tags: ['running-coach', 'rag', 'pipeline'],
  };
}

// Function to add a span to a trace
export function addSpan(trace: Trace, span: Omit<Span, 'traceId' | 'traceReferenceId'>): Trace {
  const inferredType = span.content?.type as string | undefined;
  const provider = span.content?.provider as string | undefined;
  const model = span.content?.model as string | undefined;

  const newSpan: Span = {
    ...span,
    traceReferenceId: trace.referenceId,
    sessionId: span.sessionId ?? trace.sessionId,
    // Merge attributes: span overrides trace-level defaults
    attributes: {
      ...(trace.attributes || {}),
      ...(span.attributes || {}),
      ...(inferredType ? { type: inferredType } : {}),
      ...(provider ? { provider } : {}),
      ...(model ? { model } : {}),
    },
    // Merge tags: include span name and type by default
    tags: Array.from(
      new Set([
        ...(trace.tags || []),
        ...(span.tags || []),
        span.name,
        ...(inferredType ? [inferredType] : []),
      ])
    ),
  };
  trace.spans.push(newSpan);
  return trace;
}

// Function to finalize and submit the trace
export async function submitTrace(trace: Trace): Promise<void> {
  if (!ADALINE_API_KEY) {
    console.warn("ADALINE_API_KEY is not set. Skipping trace submission.");
    return;
  }

  // Finalize the trace
  trace.endedAt = Date.now();
  // If any span has an error status, the whole trace is marked as error
  if (trace.spans.some(s => s.status === 'error')) {
    trace.status = 'error';
  }

  // Map statuses to API expected values
  const mapStatus = (s: 'success' | 'error'): 'success' | 'failure' | 'unknown' => (s === 'success' ? 'success' : 'failure');

  // Prepare payload per POST /v2/logs/trace spec
  const projectId = trace.projectId || process.env.ADALINE_PROJECT_ID || FETCHED_PROJECT_ID;
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

    // Transform content to ensure input/output are JSON strings
    let content: any = {};
    if (type) {
      content.type = type;
    }
    if (c.input !== undefined) {
      content.input = typeof c.input === 'string' ? c.input : JSON.stringify(c.input);
    }
    
    // Handle output - for Model type, special handling below; for others, embed metrics
    if (c.output !== undefined && type !== 'Model') {
      const outputObj = typeof c.output === 'object' ? c.output : { value: c.output };
      // Add cost, latency, and tokens to output if they exist on the span (for non-Model types)
      const enrichedOutput = {
        ...outputObj,
        ...(s.cost !== undefined && { cost: s.cost }),
        ...(s.latency !== undefined && { latency: s.latency }),
        ...(s.tokens && { tokens: s.tokens }),
      };
      content.output = JSON.stringify(enrichedOutput);
    } else if (c.output === undefined && type !== 'Model' && (s.cost !== undefined || s.latency !== undefined || s.tokens)) {
      // If no output but we have metrics (for non-Model types), create an output object
      content.output = JSON.stringify({
        ...(s.cost !== undefined && { cost: s.cost }),
        ...(s.latency !== undefined && { latency: s.latency }),
        ...(s.tokens && { tokens: s.tokens }),
      });
    }
    
    // For Model type, include provider, model, and cost at top-level as required
    if (type === 'Model') {
      const modelFromInput = (typeof c.input === 'object' && c.input) ? (c.input as any).model : undefined;
      content.provider = c.provider ?? 'openai';
      content.model = c.model ?? modelFromInput ?? '';
      
      // Add variables if present (per API spec for Model type)
      if (c.variables) {
        content.variables = c.variables;
      }
      
      // Add cost at content level for Model type (per API spec - only supported for Model)
      if (s.cost !== undefined) {
        content.cost = s.cost;
      }
      
      // For Model type output, preserve the output structure and add tokenUsage if available
      if (c.output !== undefined) {
        const outputObj = typeof c.output === 'object' ? c.output : JSON.parse(JSON.stringify(c.output));
        // If c.output is a string, parse it; otherwise use object directly
        const parsed = typeof outputObj === 'string' ? JSON.parse(outputObj || '{}') : outputObj;
        
        // Add tokenUsage if tokens are present on span (s.tokens comes from provider usage)
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
      content, // Content should be an object per actual API validation
      referenceId: s.referenceId,
      parentReferenceId: s.parentReferenceId,
      promptId: s.promptId,
      deploymentId: s.deploymentId,
      sessionId: s.sessionId,
      attributes: s.attributes,
      tags: s.tags,
    };
    
    // Only add runEvaluation if explicitly set
    if (s.runEvaluation !== undefined) {
      spanPayload.runEvaluation = s.runEvaluation;
    }
    
    // Don't add cost, latency, tokens at root level - they're in content
    
    return spanPayload;
  });

  try {
    const response = await fetch(`${LOGS_BASE_URL}/trace`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${ADALINE_API_KEY}`,
      },
      body: JSON.stringify({ projectId, trace: tracePayload, spans: spansPayload }),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      console.error(`Failed to submit trace to Adaline. Status: ${response.status}. Body: ${errorBody}`);
    } else {
      console.log("Trace submitted successfully to Adaline.");
    }
  } catch (error) {
    console.error("Error submitting trace to Adaline:", error);
  }
}

// Span Content Type Definitions

export interface FunctionSpanContent {
  type: 'Function';
  input: any;
  output: any;
}

export interface ModelSpanContent {
  type: 'Model';
  provider?: string;
  model?: string;
  input: {
    model: string;
    messages: any[]; // Array of message objects
  };
  output: {
    message: any; // A single message object
    metadata?: Record<string, any>;
  };
}

export interface EmbeddingsSpanContent {
    type: 'Embeddings';
    input: {
        model: string;
        texts: string[];
    };
    output: {
        embeddings: number[][];
        metadata?: Record<string, any>;
    };
}

export interface RetrievalSpanContent {
    type: 'Retrieval';
    input: {
        query: string;
        topK?: number;
    };
    output: {
        documents: any[]; // Array of document objects
    };
}
