import 'dotenv/config';
import { z } from 'zod';
import { Agent, run, tool } from '@openai/agents';
import { v4 as uuidv4 } from 'uuid';
import { nutrition_planner, weather_checker } from './tool-handler';
import { retrieveTopK, readChunkContent, parseMatchMetadata } from './retrieve';
import { getDeploymentInfo, PROMPT_ID, PROJECT_ID } from './fetchPayload';
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

export function getOrCreateTrace(): Trace {
  if (!globalTrace) {
    // Initialize a monotonic base timestamp and create the trace
    baseStartTs = now();
    globalTrace = createTrace('Agentic-RAG', PROJECT_ID, PROMPT_ID, '');
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

// Convert deployed tool to @openai/agents tool format
function createAgentTool(deployedTool: any, orchestratorRefId: string, toolExecutionPhaseRefId: string) {
  const toolName = deployedTool.definition?.schema?.name || deployedTool.function?.name || deployedTool.name;
  const toolDescription = deployedTool.definition?.schema?.description || deployedTool.function?.description || deployedTool.description || '';
  const toolParams = deployedTool.definition?.schema?.parameters || deployedTool.function?.parameters || deployedTool.parameters || {};
  
  // Create Zod schema - make all fields required to satisfy OpenAI's strict validation
  // The tool handlers can handle undefined/missing values
  const properties = toolParams.properties || {};
  const zodSchema: any = {};
  
  for (const [key, value] of Object.entries(properties)) {
    const prop = value as any;
    let fieldSchema: any = z.string();
    
    if (prop.type === 'number' || prop.type === 'integer') {
      fieldSchema = z.number();
    } else if (prop.type === 'boolean') {
      fieldSchema = z.boolean();
    } else if (prop.type === 'array') {
      fieldSchema = z.array(z.string());
    } else if (prop.type === 'object') {
      fieldSchema = z.record(z.any());
    }
    
    if (prop.description) {
      fieldSchema = fieldSchema.describe(prop.description);
    }
    
    zodSchema[key] = fieldSchema;
  }
  
  const zodObject = Object.keys(zodSchema).length > 0 ? z.object(zodSchema) : z.object({});
  
  console.log(`\n📋 Tool ${toolName}`);
  
  return tool({
    name: toolName,
    description: toolDescription,
    parameters: zodObject,
    execute: async (args: any) => {
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
            break;
          
          case 'nutrition_planner':
            result = await nutrition_planner(args);
            break;
          
          default:
            throw new Error(`Unknown tool: ${toolName}`);
        }
      } catch (error) {
        status = 'error';
        errorMessage = error instanceof Error ? error.message : String(error);
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
    },
  });
}

// Main orchestrator function using OpenAI Agents SDK
export async function orchestrateAgenticRAG(
  systemMessage: string,
  userMessage: string,
  model: string,
  deployedTools: any[],
  settings?: Record<string, any>,
  promptVariables?: Record<string, any>,
  cliOrchestratorRefId?: string
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
  // Simple token/cost estimator (approximate): 1 token ≈ 4 chars
  function estimateTokens(str: string): number {
    return Math.max(1, Math.round((str || '').length / 4));
  }
  // Optional per-model pricing (USD per 1k tokens). Unknown models get null cost.
  const pricingPerK: Record<string, { input?: number; output?: number } > = {
    'gpt-5-nano': { input: 0.05, output: 0.05 },
  };
  function estimateCost(modelName: string, inputTokens: number, outputTokens: number) {
    const p = pricingPerK[modelName];
    if (!p) return null;
    const inCost = (p.input || 0) * (inputTokens / 1000);
    const outCost = (p.output || 0) * (outputTokens / 1000);
    return Number((inCost + outCost).toFixed(6));
  }

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
    if (userRequestsRAG) {
      const ragPhaseStart = now();
      const ragPhaseRefId = uuidv4();
      let ragStatus: 'success' | 'error' = 'success';
      let ragSummary = '';
      
      console.log('\n🔍 RAG Phase...');
      
      try {
        // Pinecone query
        const pineconeStart = now();
        const matches = await retrieveTopK(5, getOrCreateTrace(), userMessage, ragPhaseRefId);
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
        for (const m of matches) {
          const { fileName, chunkNum } = await parseMatchMetadata(m);
          if (fileName && typeof chunkNum === 'number') {
            const content = await readChunkContent(fileName, chunkNum);
            lines.push(`Source: ${fileName}#${chunkNum}\n${content}`);
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
    // Include only deployed tools; RAG will be called directly when needed
    const agentTools = [
      ...deployedTools.map((tool) => createAgentTool(tool, orchestratorRefId, toolExecutionPhaseRefId)),
    ];
    console.log('\n1️⃣  Creating Agent...');

    const agent = new Agent({
      name: 'Running Coach Agent',
      model,
      instructions: finalSystemMessage,
      tools: agentTools,
    });
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
          toolsCount: agentTools.length,
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
    const result = await run(agent, userMessage);
    finalResponse = result.finalOutput || '';
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
          toolsExecuted: agentTools.length,
        },
      },
    });
    console.log(`     [SPAN] tool_execution_phase (child of agent_execution)`);
    
    // Synthesis phase removed (decorative)
    
    // Capture agent_execution end time immediately after tool phase completes
    const agentExecutionEnd = toolExecPhaseEnd;
    console.log('   ✓ Agent completed');
    
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
          toolsCount: agentTools.length,
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
    const estimatedCost = estimateCost(model, inputTokens, outputTokens);
    
    // Prepare input payload matching response.tsx format
    const inputPayload = {
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
      runEvaluation: true,
      cost: estimatedCost || undefined,
      tokens: {
        input: inputTokens,
        output: outputTokens,
        total: totalTokens,
      },
    });
    console.log(`   [SPAN] final_response (child of agent_orchestrator) - complete agent work`);

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

// CLI runner - integrates with deployed prompt
if (require.main === module) {
  (async () => {
    // Import deployed prompt assembly
    const { getInjectedMessages, DEFAULT_VARIABLES } = await import('./prompt');

    console.log('🔍 Fetching deployed prompt...');
    
    // Initialize trace
    getOrCreateTrace();
    
    const setupPhaseRefId = uuidv4();
    const setupPhaseStart = now();
    
    const fetchStart = now();
    const info = await getDeploymentInfo();
    const fetchEnd = now();
    
    // Add fetch deployment span
    addSpanToTrace({
      name: 'fetch_deployment',
      status: 'success',
      referenceId: uuidv4(),
      parentReferenceId: setupPhaseRefId,
      startedAt: fetchStart,
      endedAt: fetchEnd,
      content: {
        type: 'Function',
        input: { promptId: PROMPT_ID },
        output: {
          model: info.model,
          provider: info.providerName,
          toolsCount: info.tools.length,
        },
      },
    });
    console.log(`  [SPAN] fetch_deployment (child of setup_phase)`);
    
    const promptStart = now();
    const { systemMessage, userMessage } = await getInjectedMessages();
    const promptEnd = now();
    
    // Add prompt assembly span
    addSpanToTrace({
      name: 'prompt_assembly',
      status: 'success',
      referenceId: uuidv4(),
      parentReferenceId: setupPhaseRefId,
      startedAt: promptStart,
      endedAt: promptEnd,
      content: {
        type: 'Function',
        input: { operation: 'inject_variables' },
        output: {
          systemMessageLength: systemMessage.length,
          userMessageLength: userMessage.length,
        },
      },
    });
    console.log(`  [SPAN] prompt_assembly (child of setup_phase)`);
    
    const setupPhaseEnd = now();
    
    // Add setup_phase parent span (child of orchestrator)
    addSpanToTrace({
      name: 'setup_phase',
      status: 'success',
      referenceId: setupPhaseRefId,
      parentReferenceId: undefined, // Will be updated after orchestrator is created
      startedAt: setupPhaseStart,
      endedAt: setupPhaseEnd,
      content: {
        type: 'Function',
        input: { operation: 'initialization' },
        output: {
          deploymentFetched: true,
          promptAssembled: true,
        },
      },
    });
    console.log(`  [SPAN] setup_phase (groups fetch_deployment + prompt_assembly)`);

    console.log(`✓ Using deployed model: ${info.model}`);
    console.log(`✓ Provider: ${info.providerName}`);
    console.log(`✓ Tools: ${info.tools.length}`);

    const orchestrationStart = now();
    const orchestrationRefId = uuidv4();
    
    // Update setup_phase to have CLI orchestrator as parent
    const traceRef = getOrCreateTrace();
    const traceSpans = (traceRef as any).spans || [];
    const setupPhaseSpan = traceSpans.find((s: any) => s.referenceId === setupPhaseRefId);
    if (setupPhaseSpan) {
      setupPhaseSpan.parentReferenceId = orchestrationRefId;
    }
    
    // Use the deployed model name (OpenAI compatible) and tools
    const result = await orchestrateAgenticRAG(
      systemMessage,
      userMessage,
      info.model, // Use deployed model
      info.tools,  // Use deployed tools
      info.settings, // Use deployed settings
      DEFAULT_VARIABLES, // Pass prompt variables for observability
      orchestrationRefId // Pass parent reference for internal orchestrator
    );

    console.log('\n' + '='.repeat(80));
    console.log('📝 FINAL RESPONSE');
    console.log('='.repeat(80));
    console.log(result.finalResponse);
    console.log('='.repeat(80));
    
    // Add root CLI agentic_rag span after orchestration completes
    const orchestrationEnd = now();
    const trace = getOrCreateTrace();
    addSpanToTrace({
      name: 'agentic_rag',
      status: 'success',
      referenceId: orchestrationRefId,
      startedAt: orchestrationStart,
      endedAt: orchestrationEnd,
      content: {
        type: 'Function',
        input: {
          model: info.model,
          toolsCount: info.tools.length,
          operation: 'agentic_rag_execution',
        },
        output: {
          completed: true,
          finalResponseLength: result.finalResponse.length,
        },
      },
    });
    console.log(`  [SPAN] agentic_rag (root span)`);
    
    // Submit trace with extra safeguard normalization
    await safeSubmitTrace(trace);
  })().catch((err) => {
    console.error('Orchestrator error:', err?.message || String(err));
    process.exit(1);
  });
}
