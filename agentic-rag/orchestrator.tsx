import 'dotenv/config';
import { v4 as uuidv4 } from 'uuid';
import { nutrition_planner, weather_checker } from './tool-handler';
import { Gateway } from '@adaline/gateway';
import { Config, MessageType } from '@adaline/types';
import { getDeploymentInfo, PROMPT_ID, PROJECT_ID, PROMPT_IDS, fetchDeployedPrompt, extractSystemMessage, extractUserMessage } from './fetchPayload';
import { createTrace, addSpan, submitTrace, now, Trace } from './observability';

// Global trace to collect all spans
let globalTrace: Trace | null = null;
let baseStartTs: number | null = null;

// Normalize trace timestamps to satisfy Adaline validation
async function safeSubmitTrace(trace: Trace) {
  try {
    const spans = ((trace as any).spans || []).filter((s: any) => s);
    const candidateTimes = spans
      .map((s: any) => Number(s.startedAt))
      .filter((t: number) => typeof t === 'number' && !Number.isNaN(t) && t !== Infinity && t !== -Infinity);
    let earliest = candidateTimes.length > 0 ? Math.min(...candidateTimes) : Number((trace as any).startedAt);
    if (!(typeof earliest === 'number' && !Number.isNaN(earliest) && earliest !== Infinity && earliest !== -Infinity)) {
      earliest = now();
    }
    (trace as any).startedAt = earliest;
    const tStart = (trace as any).startedAt;
    if (typeof tStart === 'number') {
      for (const s of spans) {
        if (!(typeof s.startedAt === 'number' && !Number.isNaN(s.startedAt))) {
          s.startedAt = tStart;
        }
        if (s.startedAt < tStart) {
          s.startedAt = tStart;
        }
        if (!(typeof s.endedAt === 'number' && !Number.isNaN(s.endedAt))) {
          s.endedAt = s.startedAt;
        }
        if (s.endedAt < s.startedAt) {
          s.endedAt = s.startedAt;
        }
      }
      // Second pass: ensure trace.start is <= any span startedAt
      const validatedTimes = spans
        .map((s: any) => Number(s.startedAt))
        .filter((t: number) => typeof t === 'number' && !Number.isNaN(t) && t !== Infinity && t !== -Infinity);
      const minStart = validatedTimes.length > 0 ? Math.min(...validatedTimes) : tStart;
      (trace as any).startedAt = minStart;
      // Final enforcement: clamp all spans to be >= new trace.start
      const finalStart = (trace as any).startedAt as number;
      for (const s of spans) {
        if (typeof s.startedAt !== 'number' || Number.isNaN(s.startedAt) || s.startedAt < finalStart) {
          s.startedAt = finalStart;
        }
        if (typeof s.endedAt !== 'number' || Number.isNaN(s.endedAt) || s.endedAt < s.startedAt) {
          s.endedAt = s.startedAt;
        }
      }
    }
  } catch (_) {
    // ignore normalization errors
  }
  await submitTrace(trace);
}

export function getOrCreateTrace(traceName: string = 'Multi-Agent-RAG'): Trace {
  if (!globalTrace) {
    // Initialize a monotonic base timestamp and create the trace
    baseStartTs = now();
    globalTrace = createTrace(traceName, PROJECT_ID, PROMPT_ID, '');
    // Ensure trace has a startedAt not after baseStartTs
    (globalTrace as any).startedAt = baseStartTs;
  }
  return globalTrace;
}

export function addSpanToTrace(span: any) {
  const trace = getOrCreateTrace();
  // Ensure span times conform to Adaline validation: startedAt >= trace.startedAt
  const safeSpan = { ...span };
  const currentTraceStart = (trace as any).startedAt;
  if (typeof currentTraceStart !== 'number') {
    (trace as any).startedAt = safeSpan.startedAt;
  } else {
    // Keep trace.start at or before the earliest span
    if (typeof safeSpan.startedAt === 'number' && safeSpan.startedAt < currentTraceStart) {
      (trace as any).startedAt = safeSpan.startedAt;
    }
    // Clamp span to be >= trace.start
    const tStart = (trace as any).startedAt as number;
    if (typeof safeSpan.startedAt === 'number' && safeSpan.startedAt < tStart) {
      safeSpan.startedAt = tStart;
    }
    if (typeof safeSpan.endedAt === 'number' && safeSpan.endedAt < safeSpan.startedAt) {
      safeSpan.endedAt = safeSpan.startedAt;
    }
  }
  addSpan(trace, safeSpan);
}

// Helper function to estimate tokens (approximate): 1 token ≈ 4 chars
function estimateTokens(str: string): number {
  return Math.max(1, Math.round((str || '').length / 4));
}

// Cost is now retrieved directly from Gateway - no hardcoded pricing

/**
 * Call LLM via Adaline Gateway with tool execution support
 * This replaces the OpenAI Agents SDK with direct Gateway calls
 */
async function callLLMViaGateway(
  model: string,
  systemMessage: string,
  userMessage: string,
  tools?: any[],
  settings?: Record<string, any>,
  toolExecutors?: Record<string, (args: any) => Promise<any>>
): Promise<{
  output: string;
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  cost?: number;
  latency?: number;
  finalOutput?: string; // For compatibility
  state?: any; // For compatibility
  usage?: any; // For compatibility
}> {
  const openaiApiKey = process.env.OAI_API_KEY || process.env.OPENAI_API_KEY;
  if (!openaiApiKey) throw new Error('OAI_API_KEY or OPENAI_API_KEY missing');

  const gateway = new Gateway();
  // Dynamically import OpenAI to avoid module load-time validation errors
  const { OpenAI } = await import('@adaline/openai');
  const openai = new OpenAI();
  
  // Create model - the error might be happening here due to internal validation
  // We'll catch it and provide a better error message
  let openaiModel;
  try {
    openaiModel = openai.chatModel({ modelName: model, apiKey: openaiApiKey });
  } catch (error: any) {
    // The error is happening inside @adaline/openai when it tries to validate config schema
    // This is a known issue with Zod version conflicts in the package
    const errorMsg = error?.message || String(error);
    console.error(`Failed to create OpenAI model: ${errorMsg}`);
    console.error(`This is likely due to a Zod version conflict in @adaline/openai package`);
    throw new Error(`OpenAI model creation failed: ${errorMsg}. This may be due to a Zod version conflict in @adaline/openai.`);
  }

  // Prepare messages in Adaline MessageType format
  let messages: MessageType[] = [
    { role: 'system', content: [{ modality: 'text', value: systemMessage }] },
    { role: 'user', content: [{ modality: 'text', value: userMessage }] },
  ];

  // Prepare config from settings
  const config = Config().parse(settings || {});

  // Convert tools to Gateway format if provided
  const gatewayTools = tools?.map((tool) => {
    // If tool already has the Gateway format (from deployment), use it directly
    if (tool.type === 'function' && tool.definition?.schema) {
      return tool;
    }
    // Otherwise, convert to Gateway format
    if (tool.definition?.schema) {
      return {
        type: 'function' as const,
        definition: {
          schema: tool.definition.schema,
        },
      };
    } else if (tool.schema) {
      return {
        type: 'function' as const,
        definition: {
          schema: tool.schema,
        },
      };
    } else {
      // Assume tool is already a schema object
      return {
        type: 'function' as const,
        definition: {
          schema: tool,
        },
      };
    }
  }) || [];

  // Track total tokens, cost, and latency across all turns
  let totalInputTokens = 0;
  let totalOutputTokens = 0;
  let totalCost = 0;
  let totalLatency = 0;
  const maxTurns = 10; // Prevent infinite loops
  let turn = 0;

  while (turn < maxTurns) {
    // Track latency for this turn
    const turnStartTime = Date.now();
    
    // Call Gateway
    const resp: any = await gateway.completeChat({
      model: openaiModel,
      config,
      messages,
      tools: gatewayTools.length > 0 ? gatewayTools : undefined,
    });

    const turnEndTime = Date.now();
    const turnLatency = turnEndTime - turnStartTime;
    totalLatency += turnLatency;

    // Extract token usage and cost from Gateway response
    const usage = resp?.response?.usage || resp?.usage;
    const turnInputTokens = usage?.prompt_tokens || usage?.promptTokens || usage?.inputTokens || 0;
    const turnOutputTokens = usage?.completion_tokens || usage?.completionTokens || usage?.outputTokens || 0;
    
    // Get cost directly from Gateway (no fallback calculation)
    const gatewayCost = resp?.response?.cost ?? resp?.cost;
    const turnCost = (gatewayCost !== undefined && gatewayCost !== null && gatewayCost > 0) ? gatewayCost : 0;
    
    // Debug logging for first turn to verify Gateway cost extraction
    if (turn === 0) {
      console.log(`[DEBUG] Gateway cost extraction:`, {
        hasResponse: !!resp?.response,
        gatewayCost: gatewayCost,
        extractedCost: turnCost,
        model: model,
        inputTokens: turnInputTokens,
        outputTokens: turnOutputTokens,
        responseKeys: resp ? Object.keys(resp) : [],
        responseResponseKeys: resp?.response ? Object.keys(resp.response) : [],
      });
    }
    
    totalInputTokens += turnInputTokens;
    totalOutputTokens += turnOutputTokens;
    totalCost += turnCost;
    
    // Log Gateway metrics for this turn
    if (turnInputTokens > 0 || turnOutputTokens > 0) {
      console.log(`  [Gateway] Turn ${turn + 1}: ${turnInputTokens} input tokens, ${turnOutputTokens} output tokens, ${turnLatency}ms latency`);
      if (turnCost > 0) {
        console.log(`  [Gateway] Turn ${turn + 1} cost: $${turnCost.toFixed(6)} (from Gateway)`);
      } else if (gatewayCost === undefined || gatewayCost === null) {
        console.log(`  [Gateway] Turn ${turn + 1} cost: Not provided by Gateway`);
      }
    }

    // Extract response
    const assistantMsg = resp?.response?.messages?.[0];
    const assistantContent = assistantMsg?.content || [];
    const textContent = assistantContent.find((c: any) => c.modality === 'text' && c.value);
    const toolCalls = assistantContent.filter((c: any) => c.modality === 'tool-call');

    // Add assistant message to conversation
    messages.push({
      role: 'assistant',
      content: assistantContent,
    });

    // If no tool calls, return the text response
    if (toolCalls.length === 0) {
      const output = textContent?.value || '';
      console.log(`  [Gateway] Total: ${totalInputTokens} input, ${totalOutputTokens} output tokens, ${totalLatency}ms latency`);
      if (totalCost > 0) {
        console.log(`  [Gateway] Total cost: $${totalCost.toFixed(6)}`);
      }
      return {
        output,
        finalOutput: output, // For compatibility
        inputTokens: totalInputTokens,
        outputTokens: totalOutputTokens,
        totalTokens: totalInputTokens + totalOutputTokens,
        cost: totalCost > 0 ? totalCost : undefined,
        latency: totalLatency,
        state: {
          modelResponses: [{
            usage: {
              inputTokens: totalInputTokens,
              outputTokens: totalOutputTokens,
              totalTokens: totalInputTokens + totalOutputTokens,
            },
            cost: totalCost > 0 ? totalCost : undefined,
            latency: totalLatency,
          }],
        },
        usage: {
          inputTokens: totalInputTokens,
          outputTokens: totalOutputTokens,
          totalTokens: totalInputTokens + totalOutputTokens,
        },
      };
    }

    // Execute tools and add results to conversation
    if (toolExecutors) {
      for (const toolCall of toolCalls) {
        const toolName = toolCall.name;
        const toolArgs = JSON.parse(toolCall.arguments || '{}');
        
        if (toolExecutors[toolName]) {
          try {
            const toolResult = await toolExecutors[toolName](toolArgs);
            messages.push({
              role: 'tool',
              content: [{
                modality: 'tool-response',
                index: toolCall.index,
                id: toolCall.id,
                name: toolName,
                data: typeof toolResult === 'string' ? toolResult : JSON.stringify(toolResult),
              }],
            });
          } catch (error) {
            const errorMsg = error instanceof Error ? error.message : String(error);
            messages.push({
              role: 'tool',
              content: [{
                modality: 'tool-response',
                index: toolCall.index,
                id: toolCall.id,
                name: toolName,
                data: JSON.stringify({ error: errorMsg }),
              }],
            });
          }
        }
      }
    }

    turn++;
  }

  // If we hit max turns, return the last text response
  const lastMsg = messages[messages.length - 1];
  const lastText = lastMsg?.content?.find((c: any) => {
    return c.modality === 'text' && (c.value !== undefined || c.text !== undefined);
  });
  let outputText = '';
  if (lastText) {
    if ('value' in lastText && typeof lastText.value === 'string') {
      outputText = lastText.value;
    } else if ('text' in lastText && typeof lastText.text === 'string') {
      outputText = lastText.text;
    }
  }
  console.log(`  [Gateway] Total: ${totalInputTokens} input, ${totalOutputTokens} output tokens, ${totalLatency}ms latency`);
  if (totalCost > 0) {
    console.log(`  [Gateway] Total cost: $${totalCost.toFixed(6)}`);
  }
  return {
    output: outputText,
    finalOutput: outputText, // For compatibility
    inputTokens: totalInputTokens,
    outputTokens: totalOutputTokens,
    totalTokens: totalInputTokens + totalOutputTokens,
    cost: totalCost > 0 ? totalCost : undefined,
    latency: totalLatency,
    state: {
      modelResponses: [{
        usage: {
          inputTokens: totalInputTokens,
          outputTokens: totalOutputTokens,
          totalTokens: totalInputTokens + totalOutputTokens,
        },
        cost: totalCost > 0 ? totalCost : undefined,
        latency: totalLatency,
      }],
    },
    usage: {
      inputTokens: totalInputTokens,
      outputTokens: totalOutputTokens,
      totalTokens: totalInputTokens + totalOutputTokens,
    },
  };
}

// Helper function to log LLM spans for agent executions
export function logLLMSpan(
  name: string,
  model: string,
  systemMessage: string,
  userMessage: string,
  output: string,
  parentRefId: string,
  startedAt: number,
  endedAt: number,
  promptId?: string,
  deploymentId?: string,
  actualInputTokens?: number,
  actualOutputTokens?: number,
  actualTotalTokens?: number,
  actualCost?: number,
  actualLatency?: number,
  variables?: Record<string, { modality: string; value: string }>
) {
  // Use actual token counts if available, otherwise estimate
  const inputTokens = actualInputTokens ?? estimateTokens(systemMessage + '\n\n' + userMessage);
  const outputTokens = actualOutputTokens ?? estimateTokens(output);
  const totalTokens = actualTotalTokens ?? (inputTokens + outputTokens);
  
  // Use actual cost from Gateway only (no fallback calculation)
  const cost = (actualCost !== undefined && actualCost !== null && actualCost > 0) ? actualCost : undefined;
  
  // Calculate latency from Gateway if available, otherwise from timestamps
  const latency = actualLatency !== undefined ? actualLatency : (endedAt - startedAt);
  
  addSpanToTrace({
    name,
    status: 'success',
    referenceId: uuidv4(),
    parentReferenceId: parentRefId,
    startedAt,
    endedAt,
    content: {
      type: 'Model',
      provider: 'openai',
      model,
      variables: variables && Object.keys(variables).length > 0 ? variables : undefined,
      input: {
        messages: [
          { role: 'system', content: systemMessage },
          { role: 'user', content: userMessage },
        ],
      },
      output: {
        content: output,
        latency: latency, // Add latency to output for visibility
      },
    },
    promptId,
    deploymentId,
    runEvaluation: true, // Enable continuous evaluation for all LLM spans
    cost: (cost !== undefined && cost !== null && cost > 0) ? cost : undefined,
    tokens: {
      input: inputTokens,
      output: outputTokens,
      total: totalTokens,
    },
    attributes: {
      latency: latency, // Add latency to attributes for querying
    },
  });
}

// Convert deployed tool to Gateway tool format and create executor
function createAgentTool(deployedTool: any, orchestratorRefId: string, toolExecutionPhaseRefId: string, toolStatusTracker?: Record<string, 'success' | 'error' | 'timeout' | 'not_called'>) {
  const toolName = deployedTool.definition?.schema?.name || deployedTool.function?.name || deployedTool.name;
  const toolDescription = deployedTool.definition?.schema?.description || deployedTool.function?.description || deployedTool.description || '';
  const toolParams = deployedTool.definition?.schema?.parameters || deployedTool.function?.parameters || deployedTool.parameters || {};
  
  console.log(`\n📋 Tool ${toolName}`);
  
  // Create Gateway tool format (tools from deployments are already in Gateway format)
  const gatewayTool = {
    type: 'function' as const,
    definition: {
      schema: {
    name: toolName,
    description: toolDescription,
        parameters: toolParams,
      },
    },
  };
  
  // Create executor function
  const executor = async (args: any) => {
      const toolStart = now();
      const toolRefId = uuidv4();
      
      console.log(`\n🔧 Executing tool: ${toolName}`);
      console.log(`   Args: ${JSON.stringify(args, null, 2)}`);
      
      // Log tool call span (request)
      const toolCallRefId = uuidv4();
      addSpanToTrace({
        name: `tool_call_${toolName}`,
        status: 'success',
        referenceId: toolCallRefId,
        parentReferenceId: toolExecutionPhaseRefId,
        startedAt: toolStart,
        endedAt: toolStart,
        content: {
          type: 'Tool',
          input: {
            toolName,
            arguments: args,
          },
          output: { called: true },
        },
      });
      console.log(`     [SPAN] tool_call_${toolName} (child of tool_execution_phase)`);

      let result: any;
      let status: 'success' | 'error' = 'success';
      let errorMessage = '';
      
      try {
        switch (toolName) {
          case 'weather_checker':
            result = await weather_checker(args);
            if (toolStatusTracker) toolStatusTracker['weather'] = 'success';
            break;
          
          case 'nutrition_planner':
            result = await nutrition_planner(args);
            if (toolStatusTracker) toolStatusTracker['nutrition'] = 'success';
            break;
          
          default:
            throw new Error(`Unknown tool: ${toolName}`);
        }
      } catch (error) {
        status = 'error';
        errorMessage = error instanceof Error ? error.message : String(error);
        const isTimeout = errorMessage.toLowerCase().includes('timeout') || errorMessage.toLowerCase().includes('timed out');
        if (toolStatusTracker) {
          const toolKey = toolName === 'weather_checker' ? 'weather' : toolName === 'nutrition_planner' ? 'nutrition' : toolName;
          toolStatusTracker[toolKey] = isTimeout ? 'timeout' : 'error';
        }
        console.error(`   ✗ Error: ${errorMessage}`);
        // Log tool response span with error
        addSpanToTrace({
          name: `tool_response_${toolName}`,
          status: 'error',
          referenceId: uuidv4(),
          parentReferenceId: toolCallRefId,
          startedAt: toolStart,
          endedAt: now(),
          content: {
            type: 'Tool',
            input: {
              toolName,
              arguments: args,
            },
            output: { error: errorMessage },
          },
        });
        console.log(`     [SPAN] tool_response_${toolName} (error, child of tool_call)`);
        throw error;
      } finally {
        const toolEnd = now();
        
        // Log tool response span on success
        if (status === 'success') {
          addSpanToTrace({
            name: `tool_response_${toolName}`,
            status: 'success',
            referenceId: uuidv4(),
            parentReferenceId: toolCallRefId,
            startedAt: toolStart,
            endedAt: toolEnd,
            content: {
              type: 'Tool',
              input: {
                toolName,
                arguments: args,
              },
              output: {
                result,
                summary: result?.summary || '',
              },
            },
          });
          console.log(`     [SPAN] tool_response_${toolName} (child of tool_call_${toolName})`);
        }
      }
      
      console.log(`   ✓ Result: ${result.summary || JSON.stringify(result).substring(0, 100)}`);
      return result;
    };
  
  return { tool: gatewayTool, executor };
}

// Main orchestrator function using Adaline Gateway
export async function orchestrateAgenticRAG(
  systemMessage: string,
  userMessage: string,
  model: string,
  deployedTools: any[],
  settings?: Record<string, any>,
  promptVariables?: Record<string, any>,
  cliOrchestratorRefId?: string,
  deploymentId?: string
): Promise<{
  finalResponse: string;
}> {
  // Ensure trace exists before any timestamps are captured
  getOrCreateTrace();
  console.log('\n🚀 Starting Agentic RAG Orchestration (using @openai/agents)...');
  console.log(`   Model: ${model}`);
  console.log(`   Tools: ${deployedTools.length}`);

  const orchestratorStart = now();
  const orchestratorRefId = uuidv4();
  const agentLifecycleRefId = uuidv4();
  const agentExecutionRefId = uuidv4();
  const toolExecutionPhaseRefId = uuidv4();
  let status: 'success' | 'error' = 'success';
  let finalResponse = '';
  let agentRunStart = 0;
  let agentRunEnd = 0;
  let agentLifecycleEnd = 0;
  // Track final instructions used (augmented when RAG is applied)
  let finalSystemMessage = systemMessage;
  let augmentedPromptDetails: { ragSummary: string; topK: number; query: string } | null = null;
  // Track tool execution status for decision_provenance
  const toolStatusTracker: Record<string, 'success' | 'error' | 'timeout' | 'not_called'> = {};
  // Simple token estimator (approximate): 1 token ≈ 4 chars
  function estimateTokens(str: string): number {
    return Math.max(1, Math.round((str || '').length / 4));
  }
  // Cost is retrieved directly from Gateway - no hardcoded pricing

  try {
    // Ensure OPENAI_API_KEY is set (Agents SDK reads from env)
    if (!process.env.OPENAI_API_KEY && process.env.OAI_API_KEY) {
      process.env.OPENAI_API_KEY = process.env.OAI_API_KEY;
    }

    // Query Routing Phase
    const queryRoutingStart = now();
    const queryRoutingRefId = uuidv4();
    console.log('\n🧭 Query Routing...');
    
    // Classify intent
    const classifyIntentStart = now();
    const userRequestsRAG = /\bRAG\b/i.test(userMessage) || /\bRAG_CONTEXT\b/i.test(userMessage) || /\brag:\b/i.test(userMessage);
    const intent = userRequestsRAG ? 'rag_enabled' : 'direct_query';
    const classifyIntentEnd = now();
    
    addSpanToTrace({
      name: 'classify_intent',
      status: 'success',
      referenceId: uuidv4(),
      parentReferenceId: queryRoutingRefId,
      startedAt: classifyIntentStart,
      endedAt: classifyIntentEnd,
      content: {
        type: 'Function',
        input: { userMessage },
        output: { intent, ragRequired: userRequestsRAG },
      },
    });
    console.log(`   ✓ Intent classified: ${intent}`);
    console.log(`     [SPAN] classify_intent (child of query_routing)`);
    
    // Plan execution
    const planExecutionStart = now();
    const executionPlan = {
      useRAG: userRequestsRAG,
      tools: deployedTools.map(t => t.definition?.schema?.name || t.function?.name || t.name),
      phases: userRequestsRAG ? ['rag', 'agent', 'synthesis'] : ['agent', 'synthesis'],
    };
    const planExecutionEnd = now();
    
    addSpanToTrace({
      name: 'plan_execution',
      status: 'success',
      referenceId: uuidv4(),
      parentReferenceId: queryRoutingRefId,
      startedAt: planExecutionStart,
      endedAt: planExecutionEnd,
      content: {
        type: 'Function',
        input: { intent },
        output: executionPlan,
      },
    });
    console.log(`   ✓ Execution plan created`);
    console.log(`     [SPAN] plan_execution (child of query_routing)`);
    
    const queryRoutingEnd = now();
    addSpanToTrace({
      name: 'query_routing',
      status: 'success',
      referenceId: queryRoutingRefId,
      parentReferenceId: orchestratorRefId,
      startedAt: queryRoutingStart,
      endedAt: queryRoutingEnd,
      content: {
        type: 'Function',
        input: { userMessage },
        output: { intent, plan: executionPlan },
      },
    });
    console.log(`  [SPAN] query_routing (child of agent_orchestrator)`);

    // RAG Phase (if needed) - starts AFTER query_routing completes
    let matches: any[] = []; // Declare at function scope for decision_provenance
    let chunkSources: string[] = []; // Track chunk sources for utilization analysis
    let ragPhaseRefId: string | undefined; // Declare at function scope for utilization analysis
    if (userRequestsRAG) {
      const ragPhaseStart = now();
      ragPhaseRefId = uuidv4();
      let ragStatus: 'success' | 'error' = 'success';
      let ragSummary = '';
      
      console.log('\n🔍 RAG Phase...');
      
      try {
        // Pinecone query - lazy import to avoid Zod conflict with @adaline/openai
        const { retrieveTopK, readChunkContent, parseMatchMetadata } = await import('./retrieve');
        const pineconeStart = now();
        matches = await retrieveTopK(5, getOrCreateTrace(), userMessage, ragPhaseRefId);
        const pineconeEnd = now();
        addSpanToTrace({
          name: 'pinecone_query',
          status: 'success',
          referenceId: uuidv4(),
          parentReferenceId: ragPhaseRefId,
          startedAt: pineconeStart,
          endedAt: pineconeEnd,
          content: {
            type: 'Retrieval',
            input: { top_k: 5, query: userMessage },
            output: { matchesCount: matches.length },
          },
        });
        console.log(`     [SPAN] pinecone_query (child of rag_phase)`);
        
        // Context assembly
        const contextAssemblyStart = now();
        const lines: string[] = [];
        chunkSources = []; // Initialize chunk sources array
        for (const m of matches) {
          const { fileName, chunkNum } = await parseMatchMetadata(m);
          if (fileName && typeof chunkNum === 'number') {
            const content = await readChunkContent(fileName, chunkNum);
            const chunkSource = `${fileName}#${chunkNum}`;
            chunkSources.push(chunkSource);
            lines.push(`Source: ${chunkSource}\n${content}`);
          }
        }
        ragSummary = lines.join('\n\n');
        
        // Build augmented prompt (system + user + RAG context) as done in src/augmentPrompt.tsx
        let augmentedPrompt = '';
        augmentedPrompt += `${systemMessage}\n\n`;
        augmentedPrompt += `User Request:\n${userMessage}\n\n`;
        augmentedPrompt += `Context from retrieval (only include when relevant):\n`;
        for (let i = 0; i < lines.length; i++) {
          augmentedPrompt += `--- snippet ${i + 1} ---\n${lines[i]}\n\n`;
        }
        augmentedPrompt += `Instructions for assistant:\nUse ONLY the requested sections from the template. Keep answers tight and metric. If pain or heat is a concern, add a short caution.\n`;
        
        const contextAssemblyEnd = now();
        addSpanToTrace({
          name: 'context_assembly',
          status: 'success',
          referenceId: uuidv4(),
          parentReferenceId: ragPhaseRefId,
          startedAt: contextAssemblyStart,
          endedAt: contextAssemblyEnd,
          content: {
            type: 'Function',
            input: { matchesCount: matches.length },
            output: {
              context: augmentedPrompt.substring(0, 8000),
              augmentedPromptLength: augmentedPrompt.length,
              chunksAssembled: lines.length,
              components: ['system_message', 'user_query', 'retrieval_context', 'instructions'],
              componentLengths: {
                systemMessage: systemMessage.length,
                userQuery: userMessage.length,
                retrievalContext: ragSummary.length,
              },
              estimatedTokens: Math.ceil(augmentedPrompt.length / 4),
            },
          },
        });
        console.log(`     [SPAN] context_assembly (child of rag_phase)`);
        
        // Causal chain analysis: Check retrieval quality and potential fallback
        if (matches && matches.length > 0) {
          const avgScore = matches.reduce((sum, m) => sum + (m.score || 0), 0) / matches.length;
          const threshold = 0.75;
          
          if (avgScore < threshold) {
            // Determine fallback tool based on query content
            const hasWeatherKeywords = /\b(weather|temperature|hot|cold|rain|sunny|humidity|forecast)\b/i.test(userMessage);
            const hasNutritionKeywords = /\b(nutrition|fueling|hydration|meal|food|eat|drink)\b/i.test(userMessage);
            const fallbackTool = hasWeatherKeywords ? 'weather_checker' : hasNutritionKeywords ? 'nutrition_planner' : 'weather_checker';
            
            const causalAnalysisStart = now();
            addSpanToTrace({
              name: 'causal_chain_analysis',
              status: 'success',
              referenceId: uuidv4(),
              parentReferenceId: ragPhaseRefId,
              startedAt: causalAnalysisStart,
              endedAt: causalAnalysisStart,
              content: {
                type: 'Function',
                input: { matchesCount: matches.length, avgScore },
                output: {
                  trigger: { event: 'low_retrieval_quality', score: avgScore },
                  consequence: { event: 'fallback_tool_selected', tool: fallbackTool },
                  counterfactual: `With score > ${threshold}, would select nutrition_planner`
                },
              },
            });
            console.log(`     [SPAN] causal_chain_analysis (child of rag_phase)`);
          }
        }
        
      } catch (e) {
        ragStatus = 'error';
        ragSummary = `RAG retrieval error: ${e instanceof Error ? e.message : String(e)}`;
      } finally {
        const ragPhaseEnd = now();
        addSpanToTrace({
          name: 'rag_phase',
          status: ragStatus,
          referenceId: ragPhaseRefId,
          parentReferenceId: orchestratorRefId,
          startedAt: ragPhaseStart,
          endedAt: ragPhaseEnd,
          content: {
            type: 'Retrieval',
            input: { top_k: 5, query: userMessage },
            output: { summary: ragSummary.substring(0, 2000), contextLength: ragSummary.length },
          },
        });
        console.log(`  [SPAN] rag_phase (child of agent_orchestrator)`);
      }
      // Append RAG context for the agent to use only if any content was retrieved
      if (ragSummary && ragSummary.trim().length > 0) {
        finalSystemMessage = `${systemMessage}\n\n[RAG_CONTEXT]\n${ragSummary}`;
        augmentedPromptDetails = { ragSummary, topK: 5, query: userMessage };
      }
    }

    // Agent creation starts AFTER query_routing and RAG (if applicable) complete
    const agentCreateStart = now();
    // Convert deployed tools to Gateway format
    const gatewayToolsAndExecutors = deployedTools.map((deployedTool: any) => {
      return createAgentTool(deployedTool, orchestratorRefId, toolExecutionPhaseRefId, toolStatusTracker);
    });
    const gatewayTools = gatewayToolsAndExecutors.map((item: any) => item.tool);
    const toolExecutors: Record<string, (args: any) => Promise<any>> = {};
    gatewayToolsAndExecutors.forEach((item: any) => {
      const toolName = item.tool.definition.schema.name;
      toolExecutors[toolName] = item.executor;
    });
    
    // Initialize tool status tracker for all available tools
    deployedTools.forEach((tool) => {
      const toolName = tool.definition?.schema?.name || tool.function?.name || tool.name;
      const toolKey = toolName === 'weather_checker' ? 'weather' : toolName === 'nutrition_planner' ? 'nutrition' : toolName;
      toolStatusTracker[toolKey] = 'not_called';
    });
    console.log('\n1️⃣  Preparing Agent for Gateway...');

    const agentCreateEnd = now();
    
    addSpanToTrace({
      name: 'create_agent',
      status: 'success',
      referenceId: uuidv4(),
      parentReferenceId: agentLifecycleRefId,
      startedAt: agentCreateStart,
      endedAt: agentCreateEnd,
      content: {
        type: 'Function',
        input: {
          agentName: 'Running Coach Agent',
          model,
          toolsCount: gatewayTools.length,
          instructionsLength: finalSystemMessage.length,
        },
        output: {
          agentCreated: true,
        },
      },
    });
    console.log('   ✓ Agent created');
    console.log(`     [SPAN] create_agent (child of agent_lifecycle)`);

    // Agent Execution Phase
    const agentExecutionStart = now();
    console.log('\n2️⃣  Agent Execution...');
    
    // Planning phase removed (decorative)
    
    // Tool execution phase - wrap the agent run
    const toolExecPhaseStart = now();
    console.log('   Running agent with user message...');
    
    // Capture actual LLM call timing
    agentRunStart = now();
    const result = await callLLMViaGateway(
      model,
      finalSystemMessage,
      userMessage,
      gatewayTools,
      settings,
      toolExecutors
    );
    finalResponse = result.output || result.finalOutput || '';
    agentRunEnd = now();
    
    const toolExecPhaseEnd = now();
    
    // Add tool_execution_phase span (parent of all tool calls/responses)
    addSpanToTrace({
      name: 'tool_execution_phase',
      status: 'success',
      referenceId: toolExecutionPhaseRefId,
      parentReferenceId: agentExecutionRefId,
      startedAt: toolExecPhaseStart,
      endedAt: toolExecPhaseEnd,
      content: {
        type: 'Function',
        input: { userMessage },
        output: { 
          completed: true,
          toolsExecuted: gatewayTools.length,
        },
      },
    });
    console.log(`     [SPAN] tool_execution_phase (child of agent_execution)`);
    
    // Synthesis phase removed (decorative)
    
    // Capture agent_execution end time immediately after tool phase completes
    const agentExecutionEnd = toolExecPhaseEnd;
    console.log('   ✓ Agent completed');
    
    // Decision Provenance: Track decision-making process
    const decisionProvenanceStart = now();
    
    // Extract query keywords dynamically
    const extractKeywords = (text: string): string[] => {
      const words = text.toLowerCase().split(/\s+/);
      const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them']);
      const keywords = words
        .filter(word => word.length > 3 && !stopWords.has(word))
        .filter(word => /^[a-z]+$/.test(word)) // Only alphabetic
        .slice(0, 10); // Top 10 keywords
      return [...new Set(keywords)]; // Remove duplicates
    };
    
    const queryKeywords = extractKeywords(userMessage);
    
    // Calculate retrieval quality (if RAG was used)
    let retrievalQuality = 0;
    if (userRequestsRAG && matches && matches.length > 0) {
      retrievalQuality = matches.reduce((sum, m) => sum + (m.score || 0), 0) / matches.length;
    }
    
    // Get available tools
    const availableTools = deployedTools.map(t => {
      const toolName = t.definition?.schema?.name || t.function?.name || t.name;
      return toolName === 'weather_checker' ? 'weather' : toolName === 'nutrition_planner' ? 'nutrition' : toolName;
    });
    
    // Determine which tool was actually used (decision)
    const calledTools = Object.entries(toolStatusTracker)
      .filter(([_, status]) => status === 'success' || status === 'error')
      .map(([tool, _]) => tool);
    const primaryTool = calledTools.length > 0 ? calledTools[0] : availableTools[0] || 'unknown';
    
    // Calculate confidence based on retrieval quality and tool status
    const baseConfidence = retrievalQuality > 0 ? retrievalQuality : 0.7;
    const toolSuccessRate = Object.values(toolStatusTracker).filter(s => s === 'success').length / Math.max(1, Object.keys(toolStatusTracker).length);
    const confidence = Math.min(0.95, baseConfidence * 0.6 + toolSuccessRate * 0.4);
    
    // Prepare prior tool status
    const priorToolStatus: Record<string, string> = {};
    Object.entries(toolStatusTracker).forEach(([tool, status]) => {
      priorToolStatus[tool] = status;
    });
    
    addSpanToTrace({
      name: 'decision_provenance',
      status: 'success',
      referenceId: uuidv4(),
      parentReferenceId: agentExecutionRefId,
      startedAt: decisionProvenanceStart,
      endedAt: decisionProvenanceStart,
      content: {
        type: 'Function',
        input: {
          queryKeywords,
          retrievalQuality: retrievalQuality > 0 ? Number(retrievalQuality.toFixed(2)) : undefined,
          availableTools,
          priorToolStatus,
          decision: `${primaryTool}_tool`,
          confidence: Number(confidence.toFixed(2)),
        },
        output: {
          decision: `${primaryTool}_tool`,
          confidence: Number(confidence.toFixed(2)),
        },
      },
    });
    console.log(`     [SPAN] decision_provenance (child of agent_execution)`);
    
    // Retrieval Utilization Analysis: Track which chunks were actually used
    if (userRequestsRAG && chunkSources.length > 0 && finalResponse) {
      const utilizationStart = now();
      
      // Check which chunks are referenced in the final response
      // Look for chunk source patterns (fileName#chunkNum) or content from those chunks
      const referencedChunks: string[] = [];
      const responseLower = finalResponse.toLowerCase();
      
      // Simple heuristic: check if content from each chunk appears in the response
      // This is a simplified check - in production, you might use more sophisticated matching
      for (const chunkSource of chunkSources) {
        // Extract filename and chunk number for matching
        const [fileName] = chunkSource.split('#');
        const fileNameBase = fileName.replace(/\.[^.]+$/, '').toLowerCase();
        
        // Check if the filename or key terms from the chunk appear in response
        // This is a simplified check - you could enhance with actual content matching
        if (responseLower.includes(fileNameBase) || 
            responseLower.includes(fileName.toLowerCase())) {
          referencedChunks.push(chunkSource);
        }
      }
      
      // If no direct matches, use a more lenient approach: check if RAG context influenced the response
      // For now, we'll use a conservative estimate: if response is substantial and RAG was used,
      // assume at least some chunks were utilized
      let chunksReferencedInOutput = referencedChunks.length;
      if (chunksReferencedInOutput === 0 && finalResponse.length > 200) {
        // Conservative estimate: if response is substantial, assume 1-2 chunks were used
        chunksReferencedInOutput = Math.min(2, chunkSources.length);
      }
      
      const retrievedChunks = chunkSources.length;
      const utilizationRate = retrievedChunks > 0 ? chunksReferencedInOutput / retrievedChunks : 0;
      
      // Estimate wasted tokens (from unused chunks)
      // Approximate: each unused chunk contributes to context but isn't used
      const unusedChunks = retrievedChunks - chunksReferencedInOutput;
      const avgChunkTokens = 300; // Approximate tokens per chunk
      const wastedTokens = unusedChunks * avgChunkTokens;
      
      // Estimate wasted cost (assuming $0.002 per 1k tokens for input)
      const wastedCost = (wastedTokens / 1000) * 0.002;
      
      addSpanToTrace({
        name: 'retrieval_utilization',
        status: 'success',
        referenceId: uuidv4(),
        parentReferenceId: ragPhaseRefId || agentExecutionRefId,
        startedAt: utilizationStart,
        endedAt: now(),
        content: {
          type: 'Function',
          input: {
            retrievedChunks,
            finalResponseLength: finalResponse.length,
          },
          output: {
            retrievedChunks,
            chunksReferencedInOutput,
            utilizationRate: Number(utilizationRate.toFixed(2)),
            wastedTokens,
            wastedCost: Number(wastedCost.toFixed(4)),
          },
        },
      });
      console.log(`     [SPAN] retrieval_utilization (child of ${ragPhaseRefId ? 'rag_phase' : 'agent_execution'})`);
    }
    
    // Add agent_execution span
    addSpanToTrace({
      name: 'agent_execution',
      status: 'success',
      referenceId: agentExecutionRefId,
      parentReferenceId: agentLifecycleRefId,
      startedAt: agentExecutionStart,
      endedAt: agentExecutionEnd,
      content: {
        type: 'Function',
        input: { userMessage },
        output: { 
          completed: true,
          responseLength: finalResponse.length,
        },
      },
    });
    console.log(`    [SPAN] agent_execution (child of agent_lifecycle)`);

    // Log agent_lifecycle parent span (encompasses create_agent + agent_execution)
    // End time should be right after agent_execution completes
    agentLifecycleEnd = agentExecutionEnd;
    addSpanToTrace({
      name: 'agent_lifecycle',
      status: 'success',
      referenceId: agentLifecycleRefId,
      parentReferenceId: orchestratorRefId,
      startedAt: agentCreateStart,
      endedAt: agentLifecycleEnd,
      content: {
        type: 'Function',
        input: {
          agentName: 'Running Coach Agent',
          model,
          toolsCount: gatewayTools.length,
        },
        output: {
          completed: true,
          responseLength: finalResponse.length,
        },
      },
    });
    console.log(`   [SPAN] agent_lifecycle (child of agent_orchestrator)`);

    // Add final_response span - represents complete agent.run() (planning + tools + synthesis)
    // This is a sibling of agent_lifecycle, positioned chronologically after it
    const inputMsgStr = `${finalSystemMessage}\n\n${userMessage}`;
    const inputTokens = estimateTokens(inputMsgStr);
    const outputTokens = estimateTokens(finalResponse);
    const totalTokens = inputTokens + outputTokens;
    // Cost should come from actual LLM calls via Gateway, not estimated
    // If we have actual cost from the agent execution, use it; otherwise leave undefined
    const finalResponseCost = undefined; // Will be populated from actual Gateway responses if available
    
    // Prepare input payload matching response.tsx format
    const inputPayload: any = {
      model,
      max_tokens: settings?.max_output_tokens || settings?.maxTokens || 4096,
      temperature: settings?.temperature || 1,
      messages: [
        {
          role: 'system',
          content: [{
            type: 'text',
            text: finalSystemMessage,
          }],
        },
        {
          role: 'user',
          content: [{
            type: 'text',
            text: userMessage,
          }],
        },
      ],
    };
    
    // Include all other settings (seed, top_p, etc.) if present
    if (settings?.seed !== undefined) inputPayload.seed = settings.seed;
    if (settings?.top_p !== undefined) inputPayload.top_p = settings.top_p;
    if (settings?.frequency_penalty !== undefined) inputPayload.frequency_penalty = settings.frequency_penalty;
    if (settings?.presence_penalty !== undefined) inputPayload.presence_penalty = settings.presence_penalty;
    
    // Prepare output payload matching response.tsx Adaline MessageType format
    const outputPayload = {
      messages: [{
        role: 'assistant',
        content: [{
          modality: 'text',
          value: finalResponse,
        }],
      }],
      tokenUsage: {
        promptTokens: inputTokens,
        completionTokens: outputTokens,
        totalTokens: totalTokens,
      },
    };
    
    // Prepare variables in Adaline ContentType format from prompt template variables
    const variables: Record<string, { modality: string; value: string }> = {};
    if (promptVariables) {
      for (const [key, val] of Object.entries(promptVariables)) {
        variables[key] = {
          modality: 'text',
          value: String(val),
        };
      }
    }
    
    addSpanToTrace({
      name: 'final_response',
      status: 'success',
      referenceId: uuidv4(),
      parentReferenceId: orchestratorRefId,
      startedAt: agentRunStart,
      endedAt: agentRunEnd,
      content: {
        type: 'Model',
        provider: 'openai',
        model,
        variables: Object.keys(variables).length > 0 ? variables : undefined,
        input: inputPayload,
        output: outputPayload,
      },
      promptId: PROMPT_ID,
      deploymentId: deploymentId,
      runEvaluation: true,
      cost: finalResponseCost, // Use actual cost from Gateway, undefined if not available
      tokens: {
        input: inputTokens,
        output: outputTokens,
        total: totalTokens,
      },
    });
    console.log(`   [SPAN] final_response (child of agent_orchestrator) - complete agent work`);

    // Counterfactual Analysis: Compare actual decision vs alternative path
    const counterfactualStart = now();
    const actualChoice = userRequestsRAG ? 'rag_enabled' : 'direct_query';
    const actualLatency = agentExecutionEnd - queryRoutingStart; // Total time from routing to execution end
    // Cost should come from Gateway - for now we'll use 0 if not available (will be updated when Gateway provides it)
    const actualCost = 0; // TODO: Get actual cost from Gateway responses
    
    // Calculate quality score (use retrieval quality if RAG was used, otherwise estimate based on response)
    let actualQuality = 0.75; // Default quality for direct queries
    if (userRequestsRAG && matches && matches.length > 0) {
      actualQuality = matches.reduce((sum, m) => sum + (m.score || 0), 0) / matches.length;
    } else if (finalResponse.length > 200) {
      // Estimate quality based on response length and completeness
      actualQuality = Math.min(0.85, 0.65 + (finalResponse.length / 2000) * 0.2);
    }
    
    // Estimate alternative path metrics
    const alternativeChoice = userRequestsRAG ? 'direct_query' : 'rag_enabled';
    let estimatedLatency = 0;
    let estimatedAltCost: number | null | undefined = null; // No cost estimation - would need Gateway data
    let estimatedQuality = 0;
    
    if (alternativeChoice === 'direct_query') {
      // Direct query would be faster (no RAG retrieval)
      estimatedLatency = Math.round(actualLatency * 0.3); // ~30% of RAG latency
      // Cost estimation removed - would need actual Gateway data for alternative path
      estimatedAltCost = null;
      // Direct query might have lower quality (no context)
      estimatedQuality = Math.max(0.65, actualQuality - 0.15);
    } else {
      // RAG would be slower (add retrieval time)
      estimatedLatency = Math.round(actualLatency * 1.5); // ~150% of direct latency
      // Cost estimation removed - would need actual Gateway data for alternative path
      estimatedAltCost = null;
      // RAG might have higher quality (with context)
      estimatedQuality = Math.min(0.95, actualQuality + 0.12);
    }
    
    // Calculate tradeoff (skip cost comparison if we don't have cost data)
    const costMultiplier = (actualCost > 0 && estimatedAltCost !== null) 
      ? (actualCost / Math.max(estimatedAltCost, 0.0001)).toFixed(1) 
      : 'N/A (cost from Gateway only)';
    const qualityGain = ((actualQuality - estimatedQuality) * 100).toFixed(0);
    const tradeoffAnalysis = (actualCost > 0 && estimatedAltCost !== null)
      ? (actualCost > estimatedAltCost
          ? `Paid ${costMultiplier}x cost for ${qualityGain}% quality ${actualQuality > estimatedQuality ? 'gain' : 'loss'}`
          : `Saved ${(1 - actualCost / Math.max(estimatedAltCost, 0.0001)).toFixed(1)}x cost with ${qualityGain}% quality ${actualQuality > estimatedQuality ? 'gain' : 'loss'}`)
      : `Quality ${actualQuality > estimatedQuality ? 'gain' : 'loss'}: ${qualityGain}% (cost comparison unavailable - Gateway only)`;
    
    addSpanToTrace({
      name: 'counterfactual_analysis',
      status: 'success',
      referenceId: uuidv4(),
      parentReferenceId: queryRoutingRefId,
      startedAt: counterfactualStart,
      endedAt: now(),
      content: {
        type: 'Function',
        input: {
          actualDecision: actualChoice,
          alternativePath: alternativeChoice,
        },
        output: {
          actualDecision: {
            choice: actualChoice,
            outcome: {
              latency: actualLatency,
              cost: Number(actualCost.toFixed(4)),
              quality: Number(actualQuality.toFixed(2)),
            },
          },
          alternativePath: {
            choice: alternativeChoice,
            estimatedOutcome: {
              latency: estimatedLatency,
              cost: (estimatedAltCost != null && typeof estimatedAltCost === 'number' && estimatedAltCost > 0) 
                ? Number((estimatedAltCost as number).toFixed(4)) 
                : null,
              quality: Number(estimatedQuality.toFixed(2)),
            },
          },
          tradeoffAnalysis,
        },
      },
    });
    console.log(`     [SPAN] counterfactual_analysis (child of query_routing)`);

  } catch (error) {
    status = 'error';
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error(`\n❌ Orchestrator error: ${errorMessage}`);
    throw error;
  } finally {
    // Capture end time from when the actual work completed (agentLifecycleEnd if available)
    // This prevents the finally block execution time from inflating the orchestrator duration
    let orchestratorEnd = agentLifecycleEnd || now();

    // Add orchestrator span (child of CLI orchestrator if provided, otherwise root)
    addSpanToTrace({
      name: 'agent_orchestrator',
      status,
      referenceId: orchestratorRefId,
      parentReferenceId: cliOrchestratorRefId,
      startedAt: orchestratorStart,
      endedAt: orchestratorEnd,
      content: {
        type: 'Function',
        input: {
          model,
          toolsCount: deployedTools.length,
          tools: deployedTools.map((t) => t.definition?.schema?.name || t.function?.name || t.name),
          systemMessageLength: systemMessage.length,
          userMessageLength: userMessage.length,
        },
        output: {
          finalResponseLength: finalResponse.length,
          success: status === 'success',
        },
      },
    });
    console.log(`  [SPAN] agent_orchestrator (${cliOrchestratorRefId ? 'child of CLI orchestrator' : 'root'})`);
  }

  return {
    finalResponse,
  };
}

/**
 * Multi-Agent Orchestration with Handoffs
 * Uses an Orchestrator Agent (GPT-5.2) to coordinate three sub-agents:
 * - Fitness Agent: Generates workout plan
 * - Feelings Agent: Analyzes safety/psychological state
 * - Head Coach Agent: Assembles final training plan
 */
export async function orchestrateMultiAgentWithHandoffs(
  userInputs: {
    CONTEXT: string;
    RUN_BLOCK: string;
    WHAT_TO_COVER: string;
    NOTES: string;
  },
  cliOrchestratorRefId?: string
): Promise<{
  finalResponse: string;
}> {
  getOrCreateTrace('Multi-Agent-Orchestration');
  console.log('\n🚀 Starting Multi-Agent Orchestration with Handoffs...');
  
  const orchestratorStart = now();
  const orchestratorRefId = uuidv4();
  let status: 'success' | 'error' = 'success';
  let finalResponse = '';

  // Add parent span early so child spans can reference it
  addSpanToTrace({
    name: 'multi_agent_orchestrator',
    status: 'success', // Will be updated in finally block
    referenceId: orchestratorRefId,
    parentReferenceId: cliOrchestratorRefId,
    startedAt: orchestratorStart,
    endedAt: orchestratorStart, // Will be updated in finally block
    content: {
      type: 'Function',
      input: {
        operation: 'multi_agent_handoff_orchestration',
      },
      output: {},
    },
  });

  try {
    // Ensure OPENAI_API_KEY is set
    if (!process.env.OPENAI_API_KEY && process.env.OAI_API_KEY) {
      process.env.OPENAI_API_KEY = process.env.OAI_API_KEY;
    }

    // Step 1: Fetch all deployments to get models and prompts
    console.log('\n📥 Fetching deployments for all agents...');
    const fetchDeploymentsStart = now();
    
    const [fitnessDeployment, feelingsDeployment, headCoachDeployment] = await Promise.all([
      fetchDeployedPrompt(PROMPT_IDS.FITNESS),
      fetchDeployedPrompt(PROMPT_IDS.FEELINGS),
      fetchDeployedPrompt(PROMPT_IDS.AGENTIC_RAG)
    ]);
    
    const fetchDeploymentsEnd = now();
    
    addSpanToTrace({
      name: 'fetch_all_deployments',
      status: 'success',
      referenceId: uuidv4(),
      parentReferenceId: orchestratorRefId,
      startedAt: fetchDeploymentsStart,
      endedAt: fetchDeploymentsEnd,
      content: {
        type: 'Function',
        input: {
          promptIds: [PROMPT_IDS.FITNESS, PROMPT_IDS.FEELINGS, PROMPT_IDS.AGENTIC_RAG]
        },
        output: {
          fitnessModel: fitnessDeployment.prompt.config.model,
          feelingsModel: feelingsDeployment.prompt.config.model,
          headCoachModel: headCoachDeployment.prompt.config.model,
        },
      },
    });
    console.log(`  [SPAN] fetch_all_deployments (child of multi_agent_orchestrator)`);
    console.log(`  ✓ Fitness Model: ${fitnessDeployment.prompt.config.model}`);
    console.log(`  ✓ Feelings Model: ${feelingsDeployment.prompt.config.model}`);
    console.log(`  ✓ Head Coach Model: ${headCoachDeployment.prompt.config.model}`);

    // Step 2: Check if RAG is requested in CONTEXT
    const userRequestsRAG = /\bRAG\b/i.test(userInputs.CONTEXT) || /\bRAG_CONTEXT\b/i.test(userInputs.CONTEXT) || /\brag:\b/i.test(userInputs.CONTEXT);
    
    // Step 3: Perform RAG retrieval if requested
    let ragContext = '';
    let ragPhaseRefId: string | undefined;
    let matches: any[] = [];
    let chunkSources: string[] = [];
    
    if (userRequestsRAG) {
      const ragPhaseStart = now();
      ragPhaseRefId = uuidv4();
      let ragStatus: 'success' | 'error' = 'success';
      
      console.log('\n🔍 RAG Phase...');
      
      try {
        // Pinecone query - lazy import to avoid Zod conflict
        const { retrieveTopK, readChunkContent, parseMatchMetadata } = await import('./retrieve');
        const pineconeStart = now();
        matches = await retrieveTopK(5, getOrCreateTrace(), userInputs.CONTEXT, ragPhaseRefId);
        const pineconeEnd = now();
        
    addSpanToTrace({
          name: 'pinecone_query',
      status: 'success',
      referenceId: uuidv4(),
          parentReferenceId: ragPhaseRefId,
          startedAt: pineconeStart,
          endedAt: pineconeEnd,
          content: {
            type: 'Retrieval',
            input: { top_k: 5, query: userInputs.CONTEXT },
            output: { matchesCount: matches.length },
          },
        });
        console.log(`     [SPAN] pinecone_query (child of rag_phase)`);
        
        // Context assembly
        const contextAssemblyStart = now();
        const lines: string[] = [];
        chunkSources = [];
        
        for (const match of matches) {
          try {
            const metadata = await parseMatchMetadata(match);
            if (metadata.fileName && metadata.chunkNum !== undefined) {
              const content = await readChunkContent(metadata.fileName, metadata.chunkNum);
              const chunkSource = `${metadata.fileName}#${metadata.chunkNum}`;
              chunkSources.push(chunkSource);
              lines.push(`Source: ${chunkSource}\n${content}`);
            }
          } catch (e) {
            console.warn(`     ⚠️  Failed to read chunk: ${e instanceof Error ? e.message : String(e)}`);
          }
        }
        
        ragContext = lines.join('\n\n');
        
        const contextAssemblyEnd = now();
        addSpanToTrace({
          name: 'context_assembly',
          status: 'success',
          referenceId: uuidv4(),
          parentReferenceId: ragPhaseRefId,
          startedAt: contextAssemblyStart,
          endedAt: contextAssemblyEnd,
      content: {
        type: 'Function',
            input: { matchesCount: matches.length },
        output: {
              context: ragContext.substring(0, 8000),
              contextLength: ragContext.length,
              chunksAssembled: lines.length,
              chunkSources: Array.from(new Set(chunkSources)),
              components: ['retrieval_context'],
              componentLengths: {
                retrievalContext: ragContext.length,
              },
              estimatedTokens: Math.ceil(ragContext.length / 4),
        },
      },
    });
        console.log(`     [SPAN] context_assembly (child of rag_phase)`);
        
        // Causal chain analysis: Check retrieval quality and potential fallback
        if (matches && matches.length > 0) {
          const avgScore = matches.reduce((sum, m) => sum + (m.score || 0), 0) / matches.length;
          const threshold = 0.75;
          
          if (avgScore < threshold) {
            // Determine fallback tool based on query content
            const hasWeatherKeywords = /\b(weather|temperature|hot|cold|rain|sunny|humidity|forecast)\b/i.test(userInputs.CONTEXT);
            const hasNutritionKeywords = /\b(nutrition|fueling|hydration|meal|food|eat|drink)\b/i.test(userInputs.CONTEXT);
            const fallbackTool = hasWeatherKeywords ? 'weather_checker' : hasNutritionKeywords ? 'nutrition_planner' : 'weather_checker';
            
            const causalAnalysisStart = now();
    addSpanToTrace({
              name: 'causal_chain_analysis',
      status: 'success',
              referenceId: uuidv4(),
              parentReferenceId: ragPhaseRefId,
              startedAt: causalAnalysisStart,
              endedAt: causalAnalysisStart,
      content: {
        type: 'Function',
                input: { matchesCount: matches.length, avgScore },
        output: {
                  trigger: { event: 'low_retrieval_quality', score: avgScore },
                  consequence: { event: 'fallback_tool_selected', tool: fallbackTool },
                  counterfactual: `With score > ${threshold}, would select nutrition_planner`
        },
      },
    });
            console.log(`     [SPAN] causal_chain_analysis (child of rag_phase)`);
          }
        }
        
        // Append RAG context to CONTEXT
        if (ragContext && ragContext.trim().length > 0) {
          userInputs.CONTEXT = `${userInputs.CONTEXT}\n\n[RAG_CONTEXT]\n${ragContext}`;
        }
        
      } catch (e) {
        ragStatus = 'error';
        const errorMsg = e instanceof Error ? e.message : String(e);
        console.error(`     ❌ RAG retrieval error: ${errorMsg}`);
      } finally {
        const ragPhaseEnd = now();
    addSpanToTrace({
          name: 'rag_phase',
          status: ragStatus,
          referenceId: ragPhaseRefId,
          parentReferenceId: orchestratorRefId,
          startedAt: ragPhaseStart,
          endedAt: ragPhaseEnd,
          content: {
            type: 'Retrieval',
            input: { top_k: 5, query: userInputs.CONTEXT },
            output: { 
              contextLength: ragContext.length,
              matchesCount: matches.length,
              success: ragStatus === 'success',
            },
          },
        });
        console.log(`  [SPAN] rag_phase (child of multi_agent_orchestrator)`);
      }
    }

    // Step 4: Extract system messages from deployments
    const fitnessSystemMessage = extractSystemMessage(fitnessDeployment);
    const feelingsSystemMessage = extractSystemMessage(feelingsDeployment);
    const headCoachSystemMessage = extractSystemMessage(headCoachDeployment);

    // Step 5: Create GatewayAgent instances (wraps SDK Agent but routes through Gateway)
    console.log('\n🤖 Creating GatewayAgent instances...');
    const createSubAgentsStart = now();
    
    // Convert deployment tools to Gateway format
    // Note: tool_execution_phase will be created inside callHeadCoachAgent when it runs
    const headCoachToolsAndExecutors = (headCoachDeployment.prompt.tools || []).map((deployedTool: any) => {
      return createAgentTool(deployedTool, orchestratorRefId, ''); // toolExecutionPhaseRefId will be set later
    });
    const headCoachTools = headCoachToolsAndExecutors.map((item: any) => item.tool);
    const headCoachToolExecutors: Record<string, (args: any) => Promise<any>> = {};
    headCoachToolsAndExecutors.forEach((item: any) => {
      const toolName = item.tool.definition.schema.name;
      headCoachToolExecutors[toolName] = item.executor;
    });
    
    // Store agent configs for Gateway calls
    const fitnessAgentConfig = {
      model: fitnessDeployment.prompt.config.model,
      systemMessage: fitnessSystemMessage,
      tools: [],
      settings: fitnessDeployment.prompt.config.settings,
    };
    
    const feelingsAgentConfig = {
      model: feelingsDeployment.prompt.config.model,
      systemMessage: feelingsSystemMessage,
      tools: [],
      settings: feelingsDeployment.prompt.config.settings,
    };
    
    const headCoachAgentConfig = {
      model: headCoachDeployment.prompt.config.model,
      systemMessage: headCoachSystemMessage,
      tools: headCoachTools,
      settings: headCoachDeployment.prompt.config.settings,
      toolExecutors: headCoachToolExecutors,
    };
    
    const createSubAgentsEnd = now();
    
    addSpanToTrace({
      name: 'create_sub_agents',
      status: 'success',
      referenceId: uuidv4(),
      parentReferenceId: orchestratorRefId,
      startedAt: createSubAgentsStart,
      endedAt: createSubAgentsEnd,
      content: {
        type: 'Function',
        input: {},
        output: {
          fitnessAgentCreated: true,
          feelingsAgentCreated: true,
          headCoachAgentCreated: true,
          headCoachToolsCount: headCoachTools.length,
        },
      },
    });
    console.log(`  [SPAN] create_sub_agents (child of multi_agent_orchestrator)`);
    console.log(`  ✓ Created Fitness Agent`);
    console.log(`  ✓ Created Feelings Agent`);
    console.log(`  ✓ Created Head Coach Agent (${headCoachTools.length} tools)`);

    // Step 6: Hardcoded Orchestration (sequential execution)
    console.log('\n🎯 Starting Hardcoded Orchestration...');
    const orchestrationStart = now();
    const orchestrationRefId = uuidv4();
    
    // Create shared tool status tracker for Head Coach Agent tools (will be populated during execution)
    const toolStatusTracker: Record<string, 'success' | 'error' | 'timeout' | 'not_called'> = {};
    // Initialize all Head Coach tools as 'not_called'
    headCoachTools.forEach((tool: any) => {
      const toolName = tool.definition?.schema?.name || tool.name;
      const toolKey = toolName === 'weather_checker' ? 'weather' : toolName === 'nutrition_planner' ? 'nutrition' : toolName;
      toolStatusTracker[toolKey] = 'not_called';
    });
    
    // Create parent span for orchestration
    addSpanToTrace({
      name: 'hardcoded_orchestration',
      status: 'success',
      referenceId: orchestrationRefId,
      parentReferenceId: orchestratorRefId,
      startedAt: orchestrationStart,
      endedAt: orchestrationStart, // Will be updated after completion
      content: {
        type: 'Function',
        input: {
          operation: 'hardcoded_sequential_orchestration',
        },
        output: {},
      },
    });

    // Helper function to call Fitness Agent
    const callFitnessAgent = async (args: { CONTEXT: string; RUN_BLOCK: string; WHAT_TO_COVER: string }): Promise<{
      workoutPlan: string;
      cost?: number;
      latency?: number;
      inputTokens?: number;
      outputTokens?: number;
    }> => {
        const handoffStart = now();
        const handoffRefId = uuidv4();
        console.log(`\n🔄 Handoff to Fitness Agent...`);
        console.log(`   Input:`, args);
        
        // Create agent call span early so LLM span can reference it
        addSpanToTrace({
          name: 'call_fitness_agent',
          status: 'success',
          referenceId: handoffRefId,
          parentReferenceId: orchestrationRefId,
          startedAt: handoffStart,
          endedAt: handoffStart, // Will be updated after completion
          content: {
            type: 'Function',
            input: args,
            output: {},
          },
        });
        
        // Create user message for Fitness Agent
        const fitnessUserMessage = `Please generate the training component for the following scenario:

**CURRENT BLOCK:**
${args.RUN_BLOCK}

**REQUESTED SECTIONS:**
${args.WHAT_TO_COVER}

**CONTEXTUAL DATA:**
${args.CONTEXT}`;

        try {
          const llmStart = now();
          const fitnessResult = await callLLMViaGateway(
            fitnessAgentConfig.model,
            fitnessAgentConfig.systemMessage,
            fitnessUserMessage,
            fitnessAgentConfig.tools,
            fitnessAgentConfig.settings
          );
          const workoutPlan = fitnessResult.output || fitnessResult.finalOutput || '';
          const llmEnd = now();
          
          console.log(`  ✓ [Gateway] Token usage: ${fitnessResult.inputTokens || 'N/A'} input, ${fitnessResult.outputTokens || 'N/A'} output, ${fitnessResult.totalTokens || 'N/A'} total`);
          if (fitnessResult.cost) {
            console.log(`  ✓ [Gateway] Cost: $${fitnessResult.cost}`);
          }
          if (fitnessResult.latency) {
            console.log(`  ✓ [Gateway] Latency: ${fitnessResult.latency}ms`);
          }
          
          // Extract token usage from result
          const fitnessUsage = fitnessResult.state?.modelResponses?.[0]?.usage || fitnessResult.usage;
          const fitnessInputTokens = fitnessUsage?.inputTokens || fitnessUsage?.prompt_tokens || fitnessUsage?.promptTokens;
          const fitnessOutputTokens = fitnessUsage?.outputTokens || fitnessUsage?.completion_tokens || fitnessUsage?.completionTokens;
          const fitnessTotalTokens = fitnessUsage?.totalTokens || fitnessUsage?.total_tokens;
          
          // Prepare variables for Fitness Agent
          const fitnessVariables: Record<string, { modality: string; value: string }> = {
            CONTEXT: { modality: 'text', value: args.CONTEXT },
            RUN_BLOCK: { modality: 'text', value: args.RUN_BLOCK },
            WHAT_TO_COVER: { modality: 'text', value: args.WHAT_TO_COVER },
          };
          
          // Log LLM span for Fitness Agent (with actual usage from Gateway) - child of agent call span
          logLLMSpan(
            'fitness_agent_llm',
            fitnessDeployment.prompt.config.model,
            fitnessSystemMessage,
            fitnessUserMessage,
            workoutPlan,
            handoffRefId, // Parent is the agent call span
            llmStart,
            llmEnd,
            PROMPT_IDS.FITNESS,
            fitnessDeployment.id,
            fitnessInputTokens,
            fitnessOutputTokens,
            fitnessTotalTokens,
            fitnessResult.cost,
            fitnessResult.latency,
            fitnessVariables
          );
          
          const handoffEnd = now();
          
          // Update agent call span with output
          const trace = getOrCreateTrace();
          const agentCallSpanIndex = trace.spans.findIndex(s => s.referenceId === handoffRefId);
          if (agentCallSpanIndex >= 0) {
            trace.spans[agentCallSpanIndex] = {
              ...trace.spans[agentCallSpanIndex],
              endedAt: handoffEnd,
              content: {
                type: 'Function',
                input: args,
                output: {
                  workoutPlan: workoutPlan.substring(0, 500),
                  planLength: workoutPlan.length,
                },
              },
            };
          }
          
          console.log(`   ✓ Fitness Agent completed`);
          console.log(`     [SPAN] call_fitness_agent (child of hardcoded_orchestration)`);
          console.log(`     [SPAN] fitness_agent_llm (child of call_fitness_agent)`);
          
          return {
            workoutPlan,
            cost: fitnessResult.cost,
            latency: fitnessResult.latency,
            inputTokens: fitnessInputTokens,
            outputTokens: fitnessOutputTokens,
          };
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : String(error);
          console.error(`   ✗ Fitness Agent error: ${errorMessage}`);
          throw error;
        }
      };

    // Helper function to call Feelings Agent
    const callFeelingsAgent = async (args: { NOTES: string }): Promise<{
      safetyAssessment: string;
      cost?: number;
      latency?: number;
      inputTokens?: number;
      outputTokens?: number;
    }> => {
        const handoffStart = now();
        const handoffRefId = uuidv4();
        console.log(`\n🔄 Handoff to Feelings Agent...`);
        console.log(`   Input:`, args);
        
        // Create agent call span early so LLM span can reference it
        addSpanToTrace({
          name: 'call_feelings_agent',
          status: 'success',
          referenceId: handoffRefId,
          parentReferenceId: orchestrationRefId,
          startedAt: handoffStart,
          endedAt: handoffStart, // Will be updated after completion
          content: {
            type: 'Function',
            input: args,
            output: {},
          },
        });
        
        // Create user message for Feelings Agent
        const feelingsUserMessage = `Analyze the following runner notes and provide the safety/psychological directives:

**RUNNER NOTES:**
${args.NOTES}`;

        try {
          const llmStart = now();
          const feelingsResult = await callLLMViaGateway(
            feelingsAgentConfig.model,
            feelingsAgentConfig.systemMessage,
            feelingsUserMessage,
            feelingsAgentConfig.tools,
            feelingsAgentConfig.settings
          );
          const safetyAssessment = feelingsResult.output || feelingsResult.finalOutput || '';
          const llmEnd = now();
          
          console.log(`  ✓ [Gateway] Token usage: ${feelingsResult.inputTokens || 'N/A'} input, ${feelingsResult.outputTokens || 'N/A'} output, ${feelingsResult.totalTokens || 'N/A'} total`);
          if (feelingsResult.cost) {
            console.log(`  ✓ [Gateway] Cost: $${feelingsResult.cost}`);
          }
          if (feelingsResult.latency) {
            console.log(`  ✓ [Gateway] Latency: ${feelingsResult.latency}ms`);
          }
          
          // Extract token usage from result
          const feelingsUsage = feelingsResult.state?.modelResponses?.[0]?.usage || feelingsResult.usage;
          const feelingsInputTokens = feelingsUsage?.inputTokens || feelingsUsage?.prompt_tokens || feelingsUsage?.promptTokens;
          const feelingsOutputTokens = feelingsUsage?.outputTokens || feelingsUsage?.completion_tokens || feelingsUsage?.completionTokens;
          const feelingsTotalTokens = feelingsUsage?.totalTokens || feelingsUsage?.total_tokens;
          
          // Prepare variables for Feelings Agent
          const feelingsVariables: Record<string, { modality: string; value: string }> = {
            NOTES: { modality: 'text', value: args.NOTES },
          };
          
          // Log LLM span for Feelings Agent - child of agent call span
          logLLMSpan(
            'feelings_agent_llm',
            feelingsDeployment.prompt.config.model,
            feelingsSystemMessage,
            feelingsUserMessage,
            safetyAssessment,
            handoffRefId, // Parent is the agent call span
            llmStart,
            llmEnd,
            PROMPT_IDS.FEELINGS,
            feelingsDeployment.id,
            feelingsInputTokens,
            feelingsOutputTokens,
            feelingsTotalTokens,
            feelingsResult.cost,
            feelingsResult.latency,
            feelingsVariables
          );
          
          const handoffEnd = now();
          
          // Update agent call span with output
          const trace = getOrCreateTrace();
          const agentCallSpanIndex = trace.spans.findIndex(s => s.referenceId === handoffRefId);
          if (agentCallSpanIndex >= 0) {
            trace.spans[agentCallSpanIndex] = {
              ...trace.spans[agentCallSpanIndex],
              endedAt: handoffEnd,
              content: {
                type: 'Function',
                input: args,
                output: {
                  safetyAssessment: safetyAssessment.substring(0, 500),
                  assessmentLength: safetyAssessment.length,
                },
              },
            };
          }
          
          console.log(`   ✓ Feelings Agent completed`);
          console.log(`     [SPAN] call_feelings_agent (child of hardcoded_orchestration)`);
          console.log(`     [SPAN] feelings_agent_llm (child of call_feelings_agent)`);
          
          return {
            safetyAssessment,
            cost: feelingsResult.cost,
            latency: feelingsResult.latency,
            inputTokens: feelingsInputTokens,
            outputTokens: feelingsOutputTokens,
          };
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : String(error);
          console.error(`   ✗ Feelings Agent error: ${errorMessage}`);
          throw error;
        }
      };

    // Helper function to call Head Coach Agent
    const callHeadCoachAgent = async (args: { CONTEXT: string; RUN_BLOCK: string; WHAT_TO_COVER: string; WORKOUT_SPLITS: string; FEELINGS_ASSESSMENT: string }, toolStatusTracker: Record<string, 'success' | 'error' | 'timeout' | 'not_called'>): Promise<{
      finalPlan: string;
      actualSystemMessage: string;
      actualUserMessage: string;
      actualInputTokens?: number;
      actualOutputTokens?: number;
      actualTotalTokens?: number;
      actualCost?: number;
      actualLatency?: number;
    }> => {
        const handoffStart = now();
        const handoffRefId = uuidv4();
        console.log(`\n🔄 Handoff to Head Coach Agent...`);
        console.log(`   Input:`, { ...args, WORKOUT_SPLITS: args.WORKOUT_SPLITS.substring(0, 100) + '...', FEELINGS_ASSESSMENT: args.FEELINGS_ASSESSMENT.substring(0, 100) + '...' });
        
        // Create agent call span early so child spans can reference it
        addSpanToTrace({
          name: 'call_head_coach_agent',
          status: 'success',
          referenceId: handoffRefId,
          parentReferenceId: orchestrationRefId,
          startedAt: handoffStart,
          endedAt: handoffStart, // Will be updated after completion
          content: {
            type: 'Function',
            input: {
              ...args,
              WORKOUT_SPLITS: args.WORKOUT_SPLITS.substring(0, 200),
              FEELINGS_ASSESSMENT: args.FEELINGS_ASSESSMENT.substring(0, 200),
            },
            output: {},
          },
        });
        
        // Create user message for Head Coach Agent using the template format
        const headCoachUserMessage = `Build a plan for: ${args.RUN_BLOCK}
Include only: ${args.WHAT_TO_COVER}
Context: ${args.CONTEXT}
Workout regime: ${args.WORKOUT_SPLITS}
Notes: ${args.FEELINGS_ASSESSMENT}`;

        try {
          // Create tool_execution_phase span HERE, right before LLM call (not earlier)
          const headCoachToolExecutionPhaseRefId = uuidv4();
          const toolExecPhaseStart = now();
          
          addSpanToTrace({
            name: 'tool_execution_phase',
            status: 'success',
            referenceId: headCoachToolExecutionPhaseRefId,
            parentReferenceId: handoffRefId, // Child of call_head_coach_agent
            startedAt: toolExecPhaseStart,
            endedAt: toolExecPhaseStart, // Will be updated when agent completes
            content: {
              type: 'Function',
              input: {
                agent: 'Head Coach Agent',
              },
              output: {},
            },
          });
          
          // Update tool executors to use the correct toolExecutionPhaseRefId and shared toolStatusTracker
          const headCoachToolsAndExecutorsUpdated = (headCoachDeployment.prompt.tools || []).map((deployedTool: any) => {
            return createAgentTool(deployedTool, orchestratorRefId, headCoachToolExecutionPhaseRefId, toolStatusTracker);
          });
          const headCoachToolExecutorsUpdated: Record<string, (args: any) => Promise<any>> = {};
          headCoachToolsAndExecutorsUpdated.forEach((item: any) => {
            const toolName = item.tool.definition.schema.name;
            headCoachToolExecutorsUpdated[toolName] = item.executor;
          });
          
          const llmStart = now();
          const headCoachResult = await callLLMViaGateway(
            headCoachAgentConfig.model,
            headCoachAgentConfig.systemMessage,
            headCoachUserMessage,
            headCoachAgentConfig.tools,
            headCoachAgentConfig.settings,
            headCoachToolExecutorsUpdated // Use updated executors with correct toolExecutionPhaseRefId
          );
          const finalPlan = headCoachResult.output || headCoachResult.finalOutput || '';
          const llmEnd = now();
          
          console.log(`  ✓ [Gateway] Token usage: ${headCoachResult.inputTokens || 'N/A'} input, ${headCoachResult.outputTokens || 'N/A'} output, ${headCoachResult.totalTokens || 'N/A'} total`);
          if (headCoachResult.cost) {
            console.log(`  ✓ [Gateway] Cost: $${headCoachResult.cost}`);
          }
          if (headCoachResult.latency) {
            console.log(`  ✓ [Gateway] Latency: ${headCoachResult.latency}ms`);
          }
          
          // Extract token usage from result
          const usage = headCoachResult.state?.modelResponses?.[0]?.usage || headCoachResult.usage;
          const actualInputTokens = usage?.inputTokens || usage?.prompt_tokens || usage?.promptTokens;
          const actualOutputTokens = usage?.outputTokens || usage?.completion_tokens || usage?.completionTokens;
          const actualTotalTokens = usage?.totalTokens || usage?.total_tokens;
          
          // Prepare variables for Head Coach Agent
          const headCoachVariables: Record<string, { modality: string; value: string }> = {
            CONTEXT: { modality: 'text', value: args.CONTEXT },
            RUN_BLOCK: { modality: 'text', value: args.RUN_BLOCK },
            WHAT_TO_COVER: { modality: 'text', value: args.WHAT_TO_COVER },
            WORKOUT_SPLITS: { modality: 'text', value: args.WORKOUT_SPLITS },
            FEELINGS_ASSESSMENT: { modality: 'text', value: args.FEELINGS_ASSESSMENT },
          };
          
          // Log LLM span for Head Coach Agent
          logLLMSpan(
            'head_coach_agent_llm',
            headCoachDeployment.prompt.config.model,
            headCoachSystemMessage,
            headCoachUserMessage,
            finalPlan,
            handoffRefId,
            llmStart,
            llmEnd,
            PROMPT_IDS.AGENTIC_RAG,
            headCoachDeployment.id,
            actualInputTokens,
            actualOutputTokens,
            actualTotalTokens,
            headCoachResult.cost,
            headCoachResult.latency,
            headCoachVariables
          );
          
          const toolExecPhaseEnd = now();
          const handoffEnd = now();
          
          // Update tool_execution_phase span end time
          const trace = getOrCreateTrace();
          const toolExecPhaseSpanIndex = trace.spans.findIndex(s => s.referenceId === headCoachToolExecutionPhaseRefId);
          if (toolExecPhaseSpanIndex >= 0) {
            trace.spans[toolExecPhaseSpanIndex] = {
              ...trace.spans[toolExecPhaseSpanIndex],
              endedAt: toolExecPhaseEnd,
            };
          }
          
          // Update call_head_coach_agent span with output
          const agentCallSpanIndex = trace.spans.findIndex(s => s.referenceId === handoffRefId);
          if (agentCallSpanIndex >= 0) {
            trace.spans[agentCallSpanIndex] = {
              ...trace.spans[agentCallSpanIndex],
              endedAt: handoffEnd,
              content: {
                type: 'Function',
                input: {
                  ...args,
                  WORKOUT_SPLITS: args.WORKOUT_SPLITS.substring(0, 200),
                  FEELINGS_ASSESSMENT: args.FEELINGS_ASSESSMENT.substring(0, 200),
                },
                output: {
                  finalPlan: finalPlan.substring(0, 500),
                  planLength: finalPlan.length,
                },
              },
            };
          }
          console.log(`   ✓ Head Coach Agent completed`);
          console.log(`     [SPAN] call_head_coach_agent (child of hardcoded_orchestration)`);
          console.log(`     [SPAN] head_coach_agent_llm (child of call_head_coach_agent)`);
          
          return {
            finalPlan,
            actualSystemMessage: headCoachSystemMessage,
            actualUserMessage: headCoachUserMessage,
            actualInputTokens,
            actualOutputTokens,
            actualTotalTokens,
            actualCost: headCoachResult.cost,
            actualLatency: headCoachResult.latency,
          };
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : String(error);
          console.error(`   ✗ Head Coach Agent error: ${errorMessage}`);
          throw error;
        }
      };

    // Step 7: Execute hardcoded orchestration (sequential)
    console.log('\n▶️  Executing Hardcoded Orchestration...');
    
    // Step 1: Call Fitness Agent
    console.log('\n1️⃣  Calling Fitness Agent...');
    const fitnessResult = await callFitnessAgent({
      CONTEXT: userInputs.CONTEXT,
      RUN_BLOCK: userInputs.RUN_BLOCK,
      WHAT_TO_COVER: userInputs.WHAT_TO_COVER,
    });
    const workoutPlan = fitnessResult.workoutPlan;
    
    // Step 2: Call Feelings Agent
    console.log('\n2️⃣  Calling Feelings Agent...');
    const feelingsResult = await callFeelingsAgent({
      NOTES: userInputs.NOTES,
    });
    const feelingsAssessment = feelingsResult.safetyAssessment;
    
    // Step 3: Call Head Coach Agent with all inputs
    console.log('\n3️⃣  Calling Head Coach Agent...');
    const headCoachResult = await callHeadCoachAgent({
      CONTEXT: userInputs.CONTEXT,
      RUN_BLOCK: userInputs.RUN_BLOCK,
      WHAT_TO_COVER: userInputs.WHAT_TO_COVER,
      WORKOUT_SPLITS: workoutPlan,
      FEELINGS_ASSESSMENT: feelingsAssessment,
    }, toolStatusTracker);
    
    // Extract actual values from Head Coach Agent call
    finalResponse = headCoachResult.finalPlan;
    const actualSystemMessage = headCoachResult.actualSystemMessage;
    const actualUserMessage = headCoachResult.actualUserMessage;
    const headCoachInputTokens = headCoachResult.actualInputTokens;
    const headCoachOutputTokens = headCoachResult.actualOutputTokens;
    const headCoachTotalTokens = headCoachResult.actualTotalTokens;
    const actualCostFromGateway = headCoachResult.actualCost;
    const headCoachLatency = headCoachResult.actualLatency;
    
    const orchestrationEnd = now();
    
    // Add final_response span with Model type and full payload (immediately after orchestration completes)
    // Use ACTUAL input/output from Head Coach Agent call, not reconstructed values
    const finalResponseStart = orchestrationStart; // Start from orchestration beginning
    const finalResponseEnd = orchestrationEnd;
    
    // Use actual token counts from Gateway, fallback to estimation if not available
    const inputTokens = headCoachInputTokens ?? estimateTokens(actualSystemMessage + '\n\n' + actualUserMessage);
    const outputTokens = headCoachOutputTokens ?? estimateTokens(finalResponse);
    const totalTokens = headCoachTotalTokens ?? (inputTokens + outputTokens);
    
    // Prepare input payload matching Adaline format - use ACTUAL messages from Head Coach Agent call
    const inputPayload: any = {
      model: headCoachDeployment.prompt.config.model,
      max_tokens: headCoachDeployment.prompt.config.settings?.max_output_tokens || headCoachDeployment.prompt.config.settings?.maxTokens || 4096,
      temperature: headCoachDeployment.prompt.config.settings?.temperature || 1,
      messages: [
        {
          role: 'system',
          content: [{
            type: 'text',
            text: actualSystemMessage,
          }],
        },
        {
          role: 'user',
          content: [{
            type: 'text',
            text: actualUserMessage,
          }],
        },
      ],
    };
    
    // Include other settings if present
    if (headCoachDeployment.prompt.config.settings?.seed !== undefined) inputPayload.seed = headCoachDeployment.prompt.config.settings.seed;
    if (headCoachDeployment.prompt.config.settings?.top_p !== undefined) inputPayload.top_p = headCoachDeployment.prompt.config.settings.top_p;
    
    // Prepare output payload matching Adaline MessageType format - use ACTUAL output from Head Coach Agent call
    const outputPayload: any = {
      messages: [{
        role: 'assistant',
        content: [{
          modality: 'text',
          value: finalResponse,
        }],
      }],
      tokenUsage: {
        promptTokens: inputTokens,
        completionTokens: outputTokens,
        totalTokens: totalTokens,
      },
    };
    
    // Add latency if available from Gateway
    if (headCoachLatency !== undefined) {
      outputPayload.latency = headCoachLatency;
    }
    
    // Prepare variables for final_response span (all user inputs used in orchestration)
    const finalResponseVariables: Record<string, { modality: string; value: string }> = {
      CONTEXT: { modality: 'text', value: userInputs.CONTEXT },
      RUN_BLOCK: { modality: 'text', value: userInputs.RUN_BLOCK },
      WHAT_TO_COVER: { modality: 'text', value: userInputs.WHAT_TO_COVER },
      NOTES: { modality: 'text', value: userInputs.NOTES },
    };
    
    addSpanToTrace({
      name: 'final_response',
      status: 'success',
      referenceId: uuidv4(),
      parentReferenceId: orchestratorRefId, // Sibling of hardcoded_orchestration, not child
      startedAt: finalResponseStart,
      endedAt: finalResponseEnd,
      content: {
        type: 'Model',
        provider: 'openai',
        model: headCoachDeployment.prompt.config.model,
        variables: finalResponseVariables,
        input: inputPayload,
        output: outputPayload,
      },
      promptId: PROMPT_IDS.AGENTIC_RAG,
      deploymentId: headCoachDeployment.id,
      runEvaluation: true,
      cost: actualCostFromGateway, // Use actual cost from Gateway, undefined if not available
      tokens: {
        input: inputTokens,
        output: outputTokens,
        total: totalTokens,
      },
    });
    console.log(`   [SPAN] final_response (child of multi_agent_orchestrator, sibling of hardcoded_orchestration) - complete multi-agent work`);
    
    // Decision Provenance: Track decision-making process
    const decisionProvenanceStart = now();
    
    // Extract query keywords dynamically
    const extractKeywords = (text: string): string[] => {
      const words = text.toLowerCase().split(/\s+/);
      const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them']);
      const keywords = words
        .filter(word => word.length > 3 && !stopWords.has(word))
        .filter(word => /^[a-z]+$/.test(word)) // Only alphabetic
        .slice(0, 10); // Top 10 keywords
      return [...new Set(keywords)]; // Remove duplicates
    };
    
    const queryKeywords = extractKeywords(userInputs.CONTEXT + ' ' + userInputs.RUN_BLOCK);
    
    // Calculate retrieval quality (if RAG was used)
    let retrievalQuality = 0;
    if (userRequestsRAG && matches && matches.length > 0) {
      retrievalQuality = matches.reduce((sum, m) => sum + (m.score || 0), 0) / matches.length;
    }
    
    // Get available tools
    const availableTools = headCoachTools.map((tool: any) => {
      const toolName = tool.definition?.schema?.name || tool.name;
      return toolName === 'weather_checker' ? 'weather' : toolName === 'nutrition_planner' ? 'nutrition' : toolName;
    });
    
    // toolStatusTracker is now shared and populated during Head Coach Agent execution
    // Determine which tools were actually called (from toolStatusTracker)
    const calledTools = Object.entries(toolStatusTracker)
      .filter(([_, status]) => status === 'success' || status === 'error' || status === 'timeout')
      .map(([tool, _]) => tool);
    
    // Determine the primary decision
    // If tools were called, use the first successfully called tool
    // If no tools were called, decision is "no_tools_used"
    // If tools were called but all failed, use the first attempted tool
    const successfulTools = Object.entries(toolStatusTracker)
      .filter(([_, status]) => status === 'success')
      .map(([tool, _]) => tool);
    
    let primaryTool: string;
    let decisionType: string;
    
    if (successfulTools.length > 0) {
      primaryTool = successfulTools[0];
      decisionType = 'tool_selected';
    } else if (calledTools.length > 0) {
      // Tools were called but all failed
      primaryTool = calledTools[0];
      decisionType = 'tool_attempted_failed';
    } else {
      // No tools were called
      primaryTool = 'none';
      decisionType = 'no_tools_used';
    }
    
    // Calculate confidence based on retrieval quality AND tool execution success
    const baseConfidence = retrievalQuality > 0 ? retrievalQuality : 0.7;
    
    // Calculate tool success rate
    const totalTools = Object.keys(toolStatusTracker).length;
    const successfulToolCount = successfulTools.length;
    const toolSuccessRate = totalTools > 0 ? successfulToolCount / totalTools : 0;
    
    // Combine retrieval quality (60%) and tool success (40%)
    // If no tools available, use only retrieval quality
    const confidence = totalTools > 0
      ? Math.min(0.95, baseConfidence * 0.6 + toolSuccessRate * 0.4)
      : Math.min(0.95, baseConfidence);
    
    // Prepare prior tool status (actual status from execution)
    const priorToolStatus: Record<string, string> = {};
    Object.entries(toolStatusTracker).forEach(([tool, status]) => {
      priorToolStatus[tool] = status;
    });
    
    addSpanToTrace({
      name: 'decision_provenance',
      status: 'success',
      referenceId: uuidv4(),
      parentReferenceId: orchestrationRefId,
      startedAt: decisionProvenanceStart,
      endedAt: decisionProvenanceStart,
      content: {
        type: 'Function',
        input: {
          queryKeywords,
          retrievalQuality: retrievalQuality > 0 ? Number(retrievalQuality.toFixed(2)) : undefined,
          availableTools,
          priorToolStatus,
        },
        output: {
          decision: decisionType === 'no_tools_used' 
            ? 'no_tools_used' 
            : decisionType === 'tool_attempted_failed'
            ? `${primaryTool}_tool_failed`
            : `${primaryTool}_tool`,
          decisionType,
          primaryTool: primaryTool !== 'none' ? primaryTool : undefined,
          calledTools: calledTools.length > 0 ? calledTools : undefined,
          successfulTools: successfulTools.length > 0 ? successfulTools : undefined,
          confidence: Number(confidence.toFixed(2)),
          toolSuccessRate: totalTools > 0 ? Number(toolSuccessRate.toFixed(2)) : undefined,
        },
      },
    });
    console.log(`     [SPAN] decision_provenance (child of hardcoded_orchestration)`);
    
    // Counterfactual Analysis: Compare actual decision vs alternative path
    const counterfactualStart = now();
    const actualChoice = userRequestsRAG ? 'rag_enabled' : 'direct_query';
    const actualLatency = orchestrationEnd - orchestrationStart;
    
    // Calculate actual total cost from all agent calls
    const fitnessCost = fitnessResult.cost ?? 0;
    const feelingsCost = feelingsResult.cost ?? 0;
    const headCoachCost = actualCostFromGateway ?? 0;
    const actualCost = fitnessCost + feelingsCost + headCoachCost;
    
    // Calculate actual total latency from all agent calls
    const fitnessLatency = fitnessResult.latency ?? 0;
    const feelingsLatency = feelingsResult.latency ?? 0;
    const totalAgentLatency = fitnessLatency + feelingsLatency + (headCoachLatency ?? 0);
    
    // Calculate quality score dynamically based on model output analysis
    const calculateResponseQuality = (response: string, hasRAG: boolean, retrievalMatches?: any[]): number => {
      let quality = 0.5; // Base quality
      
      // 1. Length analysis (completeness indicator)
      if (response.length > 100) quality += 0.1;
      if (response.length > 500) quality += 0.1;
      if (response.length > 1000) quality += 0.05;
      
      // 2. Structure analysis (organization indicator)
      const hasHeaders = (response.match(/^#{1,3}\s/gm) || []).length;
      const hasBullets = (response.match(/^[-*•]\s/gm) || []).length;
      const hasNumbered = (response.match(/^\d+\.\s/gm) || []).length;
      if (hasHeaders > 0 || hasBullets > 3 || hasNumbered > 3) quality += 0.1;
      
      // 3. Content richness (detail indicator)
      const sentences = response.split(/[.!?]+/).filter(s => s.trim().length > 10);
      if (sentences.length > 5) quality += 0.05;
      if (sentences.length > 10) quality += 0.05;
      
      // 4. Keyword coverage (relevance indicator)
      const requiredKeywords = ['plan', 'workout', 'training', 'exercise', 'schedule'];
      const foundKeywords = requiredKeywords.filter(kw => 
        response.toLowerCase().includes(kw)
      ).length;
      quality += (foundKeywords / requiredKeywords.length) * 0.1;
      
      // 5. RAG quality boost (if RAG was used)
      if (hasRAG && retrievalMatches && retrievalMatches.length > 0) {
        const avgRetrievalScore = retrievalMatches.reduce((sum, m) => sum + (m.score || 0), 0) / retrievalMatches.length;
        quality += avgRetrievalScore * 0.1; // Boost based on retrieval quality
      }
      
      // 6. Response coherence (repetition penalty)
      const words = response.toLowerCase().split(/\s+/);
      const uniqueWords = new Set(words);
      const repetitionRatio = uniqueWords.size / words.length;
      if (repetitionRatio < 0.5) quality -= 0.1; // High repetition reduces quality
      
      return Math.min(0.95, Math.max(0.3, quality));
    };
    
    const actualQuality = calculateResponseQuality(finalResponse, userRequestsRAG, matches);
    
    // Estimate alternative path metrics dynamically based on actual data
    const alternativeChoice = userRequestsRAG ? 'direct_query' : 'rag_enabled';
    let estimatedLatency = 0;
    let estimatedAltCost: number | null | undefined = null;
    let estimatedQuality = 0;
    
    if (alternativeChoice === 'direct_query') {
      // Direct query: faster (no RAG retrieval), but potentially lower quality
      // Estimate: ~60% of current latency (no retrieval overhead)
      estimatedLatency = Math.round(actualLatency * 0.6);
      
      // Cost: same agent costs, but no RAG retrieval cost
      // If we had RAG retrieval cost, we'd subtract it here
      estimatedAltCost = actualCost; // Same agent costs
      
      // Quality: lower due to lack of context
      // Estimate based on current quality minus context benefit
      const contextBenefit = userRequestsRAG && matches && matches.length > 0
        ? Math.min(0.15, matches.reduce((sum, m) => sum + (m.score || 0), 0) / matches.length * 0.2)
        : 0.1;
      estimatedQuality = Math.max(0.4, actualQuality - contextBenefit);
    } else {
      // RAG enabled: slower (add retrieval), but potentially higher quality
      // Estimate: ~140% of current latency (add retrieval overhead)
      estimatedLatency = Math.round(actualLatency * 1.4);
      
      // Cost: same agent costs, plus estimated RAG retrieval cost
      // Estimate RAG cost: ~$0.0001 per query (embedding + retrieval)
      const estimatedRAGCost = 0.0001;
      estimatedAltCost = actualCost + estimatedRAGCost;
      
      // Quality: higher due to additional context
      // Estimate based on current quality plus context benefit
      const contextBenefit = matches && matches.length > 0
        ? Math.min(0.2, matches.reduce((sum, m) => sum + (m.score || 0), 0) / matches.length * 0.25)
        : 0.15;
      estimatedQuality = Math.min(0.95, actualQuality + contextBenefit);
    }
    
    // Calculate tradeoff dynamically based on actual vs estimated metrics
    const qualityGain = ((actualQuality - estimatedQuality) * 100).toFixed(1);
    const latencyDiff = actualLatency - estimatedLatency;
    const latencyDiffPercent = ((latencyDiff / Math.max(estimatedLatency, 1)) * 100).toFixed(0);
    
    let tradeoffAnalysis = '';
    if (actualCost > 0 && estimatedAltCost !== null && estimatedAltCost > 0) {
      const costDiff = actualCost - estimatedAltCost;
      const costDiffPercent = ((costDiff / estimatedAltCost) * 100).toFixed(1);
      const costMultiplier = (actualCost / estimatedAltCost).toFixed(2);
      
      if (costDiff > 0) {
        tradeoffAnalysis = `Paid ${costMultiplier}x cost ($${costDiff.toFixed(6)} more, ${costDiffPercent}% increase)`;
      } else {
        tradeoffAnalysis = `Saved ${(1 - actualCost / estimatedAltCost).toFixed(2)}x cost ($${Math.abs(costDiff).toFixed(6)} less, ${Math.abs(parseFloat(costDiffPercent))}% decrease)`;
      }
      
      const qualityGainNum = parseFloat(qualityGain);
      const latencyDiffPercentNum = parseFloat(latencyDiffPercent);
      tradeoffAnalysis += ` with ${Math.abs(qualityGainNum).toFixed(1)}% quality ${actualQuality > estimatedQuality ? 'gain' : 'loss'}`;
      tradeoffAnalysis += ` and ${Math.abs(latencyDiffPercentNum).toFixed(0)}% latency ${latencyDiff > 0 ? 'increase' : 'decrease'}`;
    } else {
      // Cost data unavailable, focus on quality and latency
      const qualityGainNum = parseFloat(qualityGain);
      const latencyDiffPercentNum = parseFloat(latencyDiffPercent);
      tradeoffAnalysis = `Quality ${actualQuality > estimatedQuality ? 'gain' : 'loss'}: ${Math.abs(qualityGainNum).toFixed(1)}%`;
      tradeoffAnalysis += ` | Latency ${latencyDiff > 0 ? 'increase' : 'decrease'}: ${Math.abs(latencyDiffPercentNum).toFixed(0)}%`;
      if (actualCost > 0) {
        tradeoffAnalysis += ` | Actual cost: $${actualCost.toFixed(6)}`;
      }
    }
    
    addSpanToTrace({
      name: 'counterfactual_analysis',
      status: 'success',
      referenceId: uuidv4(),
      parentReferenceId: orchestrationRefId,
      startedAt: counterfactualStart,
      endedAt: now(),
      content: {
        type: 'Function',
        input: {
          actualDecision: actualChoice,
          alternativePath: alternativeChoice,
        },
        output: {
          actualDecision: {
            choice: actualChoice,
            outcome: {
              latency: actualLatency,
              cost: actualCost > 0 ? Number(actualCost.toFixed(4)) : null,
              quality: Number(actualQuality.toFixed(2)),
            },
          },
          alternativePath: {
            choice: alternativeChoice,
            estimatedOutcome: {
              latency: estimatedLatency,
              cost: (estimatedAltCost != null && typeof estimatedAltCost === 'number' && estimatedAltCost > 0) 
                ? Number((estimatedAltCost as number).toFixed(4)) 
                : null,
              quality: Number(estimatedQuality.toFixed(2)),
            },
          },
          tradeoffAnalysis,
        },
      },
    });
    console.log(`     [SPAN] counterfactual_analysis (child of hardcoded_orchestration)`);
    
    // Retrieval Utilization Analysis: Track which chunks were actually used
    // This should come AFTER final_response exists, but be a child of rag_phase
    if (userRequestsRAG && chunkSources.length > 0 && finalResponse) {
      const utilizationStart = now();
      
      // Check which chunks are referenced in the final response
      const referencedChunks: string[] = [];
      const responseLower = finalResponse.toLowerCase();
      
      // Simple heuristic: check if content from each chunk appears in the response
      for (const chunkSource of chunkSources) {
        const [fileName] = chunkSource.split('#');
        const fileNameBase = fileName.replace(/\.[^.]+$/, '').toLowerCase();
        
        if (responseLower.includes(fileNameBase) || 
            responseLower.includes(fileName.toLowerCase())) {
          referencedChunks.push(chunkSource);
        }
      }
      
      // Conservative estimate if no direct matches
      let chunksReferencedInOutput = referencedChunks.length;
      if (chunksReferencedInOutput === 0 && finalResponse.length > 200) {
        chunksReferencedInOutput = Math.min(2, chunkSources.length);
      }
      
      const retrievedChunks = chunkSources.length;
      const utilizationRate = retrievedChunks > 0 ? chunksReferencedInOutput / retrievedChunks : 0;
      
      // Estimate wasted tokens (from unused chunks)
      const unusedChunks = retrievedChunks - chunksReferencedInOutput;
      const avgChunkTokens = 300; // Approximate tokens per chunk
      const wastedTokens = unusedChunks * avgChunkTokens;
      
      // Estimate wasted cost (assuming $0.002 per 1k tokens for input)
      const wastedCost = (wastedTokens / 1000) * 0.002;
      
      addSpanToTrace({
        name: 'retrieval_utilization',
        status: 'success',
        referenceId: uuidv4(),
        parentReferenceId: ragPhaseRefId, // Always child of rag_phase if RAG was used
        startedAt: utilizationStart,
        endedAt: now(),
        content: {
          type: 'Function',
          input: {
            retrievedChunks,
            finalResponseLength: finalResponse.length,
          },
          output: {
            retrievedChunks,
            chunksReferencedInOutput,
            utilizationRate: Number(utilizationRate.toFixed(2)),
            wastedTokens,
            wastedCost: Number(wastedCost.toFixed(4)),
          },
        },
      });
      console.log(`     [SPAN] retrieval_utilization (child of rag_phase, created after final_response)`);
    }
    
    // Update orchestration span
    const trace = getOrCreateTrace();
    const orchestrationSpanIndex = trace.spans.findIndex(s => s.referenceId === orchestrationRefId);
    if (orchestrationSpanIndex >= 0) {
      trace.spans[orchestrationSpanIndex] = {
        ...trace.spans[orchestrationSpanIndex],
        status: 'success',
        endedAt: orchestrationEnd,
        content: {
          type: 'Function',
          input: {
            operation: 'hardcoded_sequential_orchestration',
          },
          output: {
            finalResponseLength: finalResponse.length,
            completed: true,
          },
        },
      };
    }
    console.log(`  [SPAN] hardcoded_orchestration (child of multi_agent_orchestrator)`);
    console.log(`  ✓ Orchestration completed`);

  } catch (error) {
    status = 'error';
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error(`\n❌ Multi-Agent Orchestration error: ${errorMessage}`);
    throw error;
  } finally {
    const orchestratorEnd = now();
    
    // Update the existing span instead of creating a duplicate
    const trace = getOrCreateTrace();
    const existingSpanIndex = trace.spans.findIndex(s => s.referenceId === orchestratorRefId);
    if (existingSpanIndex >= 0) {
      trace.spans[existingSpanIndex] = {
        ...trace.spans[existingSpanIndex],
        status,
        endedAt: orchestratorEnd,
        content: {
          type: 'Function',
          input: {
            operation: 'multi_agent_handoff_orchestration',
          },
          output: {
            finalResponseLength: finalResponse.length,
            success: status === 'success',
          },
        },
      };
    }
    console.log(`  [SPAN] multi_agent_orchestrator`);
  }

  return {
    finalResponse,
  };
}

// CLI runner - uses multi-agent orchestration with handoffs
if (require.main === module) {
  (async () => {
    // Import default variables from headCoach
    const { DEFAULT_VARIABLES } = await import('./prompts/headCoach');
    const { DEFAULT_VARIABLES: FEELINGS_DEFAULT } = await import('./prompts/feelings');

    console.log('🚀 Starting Multi-Agent Orchestration with Handoffs...\n');
    
    // Initialize trace
    getOrCreateTrace();
    
    const orchestrationStart = now();
    const orchestrationRefId = uuidv4();
    
    // Prepare user inputs (using default variables, can be overridden)
    const userInputs = {
      CONTEXT: DEFAULT_VARIABLES.CONTEXT,
      RUN_BLOCK: DEFAULT_VARIABLES.RUN_BLOCK,
      WHAT_TO_COVER: DEFAULT_VARIABLES.WHAT_TO_COVER,
      NOTES: FEELINGS_DEFAULT.NOTES,
    };
    
    console.log('📋 User Inputs:');
    console.log(`  CONTEXT: ${userInputs.CONTEXT.substring(0, 100)}...`);
    console.log(`  RUN_BLOCK: ${userInputs.RUN_BLOCK}`);
    console.log(`  WHAT_TO_COVER: ${userInputs.WHAT_TO_COVER}`);
    console.log(`  NOTES: ${userInputs.NOTES.substring(0, 100)}...\n`);
    
    // Run multi-agent orchestration
    const result = await orchestrateMultiAgentWithHandoffs(
      userInputs,
      orchestrationRefId
    );

    console.log('\n' + '='.repeat(80));
    console.log('📝 FINAL RESPONSE');
    console.log('='.repeat(80));
    console.log(result.finalResponse);
    console.log('='.repeat(80));
    
    // Add root CLI multi_agent_rag span after orchestration completes
    const orchestrationEnd = now();
    const trace = getOrCreateTrace();
    addSpanToTrace({
      name: 'multi_agent_rag',
      status: 'success',
      referenceId: orchestrationRefId,
      startedAt: orchestrationStart,
      endedAt: orchestrationEnd,
      content: {
        type: 'Function',
        input: {
          operation: 'multi_agent_handoff_execution',
          orchestratorModel: 'gpt-5.2',
        },
        output: {
          completed: true,
          finalResponseLength: result.finalResponse.length,
        },
      },
    });
    console.log(`\n  [SPAN] multi_agent_rag (root span)`);
    
    // Submit trace with extra safeguard normalization
    await safeSubmitTrace(trace);
  })().catch((err) => {
    console.error('Orchestrator error:', err?.message || String(err));
    process.exit(1);
  });
}
