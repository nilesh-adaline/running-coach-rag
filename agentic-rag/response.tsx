import 'dotenv/config';
import { Gateway } from '@adaline/gateway';
import { Config, MessageType } from '@adaline/types';
import { v4 as uuidv4 } from 'uuid';
import { getDeploymentInfo, getLatestDeployedPrompt, PROMPT_ID, PROJECT_ID } from './fetchPayload';
import { createTrace, addSpan, submitTrace, now, Trace } from './observability';
import { nutrition_planner, weather_checker } from './tool-handler';
import { getInjectedMessages } from './prompts/headCoach';
import { retrieveTopK, readChunkContent, parseMatchMetadata } from './retrieve';

// Global trace to collect all spans
let globalTrace: Trace | null = null;

export function getOrCreateTrace(): Trace {
  if (!globalTrace) {
    const deployed = getLatestDeployedPrompt();
    globalTrace = createTrace('Agentic-RAG', PROJECT_ID, PROMPT_ID, '');
  }
  return globalTrace;
}

export function addSpanToTrace(span: any) {
  const trace = getOrCreateTrace();
  addSpan(trace, span);
}

async function main() {
  // Use OpenAI API key for the OpenAI adapter
  const openaiApiKey = process.env.OAI_API_KEY;
  if (!openaiApiKey) throw new Error('OAI_API_KEY missing');

  // Initialize the trace first
  getOrCreateTrace();

  // 1) Fetch deployment info (provider, model, settings, tools)
  const fetchStart = now();
  const info = await getDeploymentInfo();
  const fetchEnd = now();
  
  // Add fetch span
  addSpanToTrace({
    name: 'fetch_payload',
    status: 'success',
    referenceId: uuidv4(),
    startedAt: fetchStart,
    endedAt: fetchEnd,
    content: {
      type: 'Function',
      input: { promptId: PROMPT_ID, projectId: PROJECT_ID },
      output: {
        providerName: info.providerName,
        model: info.model,
        toolsCount: info.tools.length,
      },
    },
  });
  console.log(`  [SPAN 1] fetch_payload`);
  
  const providerName = info.providerName;
  const modelName = info.model;
  // Use the deployed model from fetchPayload for both tool-call and final response
  const settings = info.settings || {};
  const tools = info.tools || [];
  const config = Config().parse(settings);

  // 2) Build final system + user messages with variable injection
  const promptStart = now();
  const { systemMessage, userMessage } = await getInjectedMessages();
  const promptEnd = now();
  
  // Add prompt assembly span
  addSpanToTrace({
    name: 'prompt_assembly',
    status: 'success',
    referenceId: uuidv4(),
    startedAt: promptStart,
    endedAt: promptEnd,
    content: {
      type: 'Function',
      input: { systemMessageLen: systemMessage.length, userMessageLen: userMessage.length },
      output: { assembled: true },
    },
  });
  console.log(`  [SPAN 2] prompt_assembly`);
  
  // 3) Assemble augmented prompt with retrieval (parent span with children)
  // Ensure augmented prompt starts strictly after prompt_assembly
  let augmentedParentStart = now();
  if (augmentedParentStart <= promptEnd) {
    augmentedParentStart = promptEnd + 1;
  }
  const augmentedParentRefId = uuidv4();
  const snippets: string[] = [];
  const fullPrompt = `${systemMessage}\n\nUser request:\n${userMessage}`;
  const matches = await retrieveTopK(5, getOrCreateTrace(), fullPrompt, augmentedParentRefId);
  for (let i = 0; i < matches.length; i++) {
    const m = matches[i];
    const { fileName, chunkNum } = await parseMatchMetadata(m);
    if (fileName && typeof chunkNum === 'number') {
      const content = await readChunkContent(fileName, chunkNum);
      if (content && content.trim()) snippets.push(content);
    }
  }

  // Ensure augmentation child starts after retrieval and after parent start
  let augmentationStart = now();
  if (augmentationStart <= augmentedParentStart) {
    augmentationStart = augmentedParentStart + 1;
  }
  let augmentedSystem = systemMessage;
  if (snippets.length > 0) {
    augmentedSystem += `\n\nContext from knowledge base (use only when relevant):\n`;
    for (let i = 0; i < snippets.length; i++) {
      augmentedSystem += `--- snippet ${i + 1} ---\n${snippets[i]}\n\n`;
    }
  }
  const augmentedUser = `${userMessage}\n\nInstructions: Use ONLY relevant context. Be concise and metric.`;
  let augmentationEnd = now();
  if (augmentationEnd <= augmentationStart) {
    augmentationEnd = augmentationStart + 1;
  }

  addSpanToTrace({
    name: 'prompt_augmentation',
    status: 'success',
    parentReferenceId: augmentedParentRefId,
    startedAt: augmentationStart,
    endedAt: augmentationEnd,
    content: {
      type: 'Function',
      input: {
        operation: 'augment_prompt_with_retrieval_context',
        coachTemplate: systemMessage,
        userQuery: userMessage,
        snippetsIncluded: snippets.length,
      },
      output: {
        systemMessage: augmentedSystem,
        userMessage: augmentedUser,
        components: ['coach_template','user_query','retrieval_context','instructions'],
      },
    },
  });

  let augmentedParentEnd = now();
  if (augmentedParentEnd <= augmentationEnd) {
    augmentedParentEnd = augmentationEnd + 1;
  }
  addSpanToTrace({
    name: 'assemble_augmented_prompt',
    status: 'success',
    referenceId: augmentedParentRefId,
    startedAt: augmentedParentStart,
    endedAt: augmentedParentEnd,
    content: {
      type: 'Function',
      input: { topK: 5 },
      output: { snippetsRetrieved: snippets.length, matchesFound: matches.length },
    },
  });
  
  // Use augmented messages for the LLM call
  const finalSystem = augmentedSystem;
  const finalUser = augmentedUser;
  
  const responseStart = now();

  // 3) Prepare messages in Adaline MessageType format
  const messages: MessageType[] = [
    { role: 'system', content: [{ modality: 'text', value: finalSystem }] },
    { role: 'user', content: [{ modality: 'text', value: finalUser }] },
  ];

  // 4) Create Gateway and OpenAI chat model adapter
  const gateway = new Gateway();
  const { OpenAI } = await import('@adaline/openai');
  const openai = new OpenAI();
  const openaiModel = openai.chatModel({ modelName, apiKey: openaiApiKey });

  console.log(`Calling LLM via Adaline Gateway...`);
  console.log(`  Provider: ${providerName}`);
  console.log(`  Model:    ${modelName}`);

  // Response parent span reference ID
  const responseRefId = uuidv4();

  // 5) Execute chat completion
  const modelStart = now();
  const resp: any = await gateway.completeChat({
    model: openaiModel,
    config,
    messages,
    tools,
  });
  const modelEnd = now();

  // 6) Extract and log assistant response
  const assistantMsg = resp?.response?.messages?.[0];
  const textContent = assistantMsg?.content?.find((c: any) => c.modality === 'text' && c.value);
  const assistantText = textContent?.value || '';

  console.log('\n=== Assistant Response ===\n');
  console.log(assistantText || '[No text returned]');

  // Debug: log raw response for local inspection
  console.log('\n=== DEBUG: Raw Gateway Response ===');
  try {
    console.log(JSON.stringify(resp?.response ?? resp, null, 2));
  } catch {
    console.log(resp?.response ?? resp);
  }

  // If no text returned but we have tool-calls, handle them locally and do a follow-up call.
  const assistantContent = assistantMsg?.content || [];
  let toolCalls = assistantContent.filter((c: any) => c.modality === 'tool-call');
  
  let responseEnd = now();
  
  // If the model didn't produce tool-calls, synthesize two: weather_checker and nutrition_planner
  if (!assistantText && toolCalls.length === 0) {
    const syntheticWeatherArgs = {
      location: 'Runner City, Country',
      datetime: new Date().toISOString(),
      duration_minutes: 60,
    };
    const syntheticNutritionArgs = {
      run_block: '60-minute recovery run on flat surface; easy pace',
      what_to_cover: 'hydration & electrolytes',
      context: 'Cool ~15°C; water every 3 km; prior ankle sprain; avoid uneven terrain',
    };
    toolCalls = [
      { modality: 'tool-call', name: 'weather_checker', id: 'synthetic_weather_1', index: 0, arguments: JSON.stringify(syntheticWeatherArgs) },
      { modality: 'tool-call', name: 'nutrition_planner', id: 'synthetic_nutrition_1', index: 1, arguments: JSON.stringify(syntheticNutritionArgs) },
    ];
  }

  if (!assistantText && toolCalls.length > 0) {
    // Execute all tool calls and collect results
    const toolResults: any[] = [];
    
    for (const toolCall of toolCalls) {
      const name = toolCall?.name;
      let args: any = {};
      try {
        args = toolCall?.arguments ? JSON.parse(toolCall.arguments) : {};
      } catch {
        args = {};
      }

      let toolResult: any = null;
      let toolStart = now();
      
      if (name === 'nutrition_planner') {
        toolResult = await nutrition_planner(args);
      } else if (name === 'weather_checker') {
        toolResult = await weather_checker(args);
      }
      let toolEnd = now();
      if (toolEnd <= toolStart) toolEnd = toolStart + 1;
      
      if (toolResult) {
        toolResults.push({ toolCall, toolResult, toolStart, toolEnd });
      }
    }

    if (toolResults.length > 0) {
      // Build a follow-up conversation: include prior system/user, assistant tool-calls, and all tool results
      const followUpMessages: MessageType[] = [
        { role: 'system', content: [{ modality: 'text', value: systemMessage }] },
        { role: 'user', content: [{ modality: 'text', value: userMessage }] },
        // Include all assistant tool-calls
        { role: 'assistant', content: toolCalls },
      ];
      
      // Add each tool response as a separate message (Gateway requires one content item per tool message)
      toolResults.forEach((tr, idx) => {
        followUpMessages.push({
          role: 'tool',
          content: [{
            modality: 'tool-response' as const,
            name: tr.toolCall.name,
            id: tr.toolCall.id || `local_tool_result_${idx + 1}`,
            data: JSON.stringify(tr.toolResult),
            index: idx
          }]
        });
      });

      // Reuse the same deployed model for the follow-up completion
      const followUpOpenaiModel = openai.chatModel({ modelName, apiKey: openaiApiKey });

      console.log(`\nCalling follow-up LLM (${modelName}) with tool result...`);
      let finalStart = now();
      // Ensure final response starts after all tool responses
      const latestToolEnd = toolResults.reduce((m, tr) => Math.max(m, tr.toolEnd), modelEnd);
      if (finalStart <= latestToolEnd) finalStart = latestToolEnd + 1;
      const followUpResp: any = await gateway.completeChat({
        model: followUpOpenaiModel,
        config,
        messages: followUpMessages,
      });
      let finalEnd = now();
      if (finalEnd <= finalStart) finalEnd = finalStart + 1;
      const followMsg = followUpResp?.response?.messages?.[0];
      const followText = followMsg?.content?.find((c: any) => c.modality === 'text' && c.value)?.value;
      console.log('\n=== Final Assistant Response (after tool) ===\n');
      console.log(followText || '[No text returned after tool]');
      
      responseEnd = now();
      if (responseEnd <= finalEnd) responseEnd = finalEnd + 1;
      
      // Add response parent span marker
      console.log(`  [SPAN 4] response (parent)`);

      // Add distinct tool_call spans for each tool requested by the assistant
      toolCalls.forEach((tc: any, idx: number) => {
        addSpanToTrace({
          name: `tool_call_${tc.name}`,
          status: 'success',
          referenceId: uuidv4(),
          parentReferenceId: responseRefId,
          startedAt: modelStart,
          endedAt: modelEnd,
          content: {
            type: 'Model',
            provider: providerName,
            model: modelName,
            input: { messages, tools, toolRequested: tc.name, arguments: tc.arguments },
            output: resp?.response?.messages?.[0],
          },
          tokens: {
            input: resp?.response?.usage?.promptTokens ?? resp?.response?.usage?.prompt_tokens ?? 0,
            output: resp?.response?.usage?.completionTokens ?? resp?.response?.usage?.completion_tokens ?? 0,
            total: resp?.response?.usage?.totalTokens ?? resp?.response?.usage?.total_tokens ?? 0,
          },
        });
        console.log(`    [SPAN 4a.${idx + 1}] tool_call_${tc.name} (child)`);
      });
      
      // Add individual tool_response spans for each tool (ensure they log before final_response)
      toolResults.forEach((tr, idx) => {
        addSpanToTrace({
          name: `tool_response_${tr.toolResult.name}`,
          status: 'success',
          referenceId: uuidv4(),
          parentReferenceId: responseRefId,
          startedAt: tr.toolStart,
          endedAt: tr.toolEnd,
          content: {
            type: 'Function',
            input: { toolCall: tr.toolCall },
            output: { toolResult: tr.toolResult },
          },
        });
        console.log(`    [SPAN 4b.${idx + 1}] tool_response_${tr.toolResult.name} (child)`);
      });
      
      addSpanToTrace({
        name: 'final_response',
        status: 'success',
        referenceId: uuidv4(),
        parentReferenceId: responseRefId,
        startedAt: finalStart,
        endedAt: finalEnd,
        content: {
          type: 'Model',
          provider: providerName,
          model: modelName,
          input: { messages: followUpMessages },
          output: followMsg,
        },
        tokens: {
          input: followUpResp?.response?.usage?.promptTokens ?? followUpResp?.response?.usage?.prompt_tokens ?? 0,
          output: followUpResp?.response?.usage?.completionTokens ?? followUpResp?.response?.usage?.completion_tokens ?? 0,
          total: followUpResp?.response?.usage?.totalTokens ?? followUpResp?.response?.usage?.total_tokens ?? 0,
        },
      });
      console.log(`    [SPAN 4c] final_response (child)`);
      
      // Add parent response span
      addSpanToTrace({
        name: 'response',
        status: 'success',
        referenceId: responseRefId,
        startedAt: responseStart,
        endedAt: responseEnd,
        content: {
          type: 'Function',
          input: { messagesCount: messages.length, toolsCount: tools.length },
          output: { finalText: followText },
        },
      });
    }
  } else {
    // Direct response without tool calls - still create response span
    console.log(`  [SPAN 4] response (parent)`);
    addSpanToTrace({
      name: 'direct_response',
      status: 'success',
      referenceId: uuidv4(),
      parentReferenceId: responseRefId,
      startedAt: modelStart,
      endedAt: modelEnd,
      content: {
        type: 'Model',
        provider: providerName,
        model: modelName,
        input: { messages, tools },
        output: assistantMsg,
      },
      tokens: {
        input: resp?.response?.usage?.promptTokens ?? resp?.response?.usage?.prompt_tokens ?? 0,
        output: resp?.response?.usage?.completionTokens ?? resp?.response?.usage?.completion_tokens ?? 0,
        total: resp?.response?.usage?.totalTokens ?? resp?.response?.usage?.total_tokens ?? 0,
      },
    });
    console.log(`    [SPAN 4a] direct_response (child)`);
    
    // Add parent response span
    addSpanToTrace({
      name: 'response',
      status: 'success',
      referenceId: responseRefId,
      startedAt: responseStart,
      endedAt: responseEnd,
      content: {
        type: 'Function',
        input: { messagesCount: messages.length, toolsCount: tools.length },
        output: { responseText: assistantText },
      },
    });
  }

  // 7) Log basic usage/cost if present
  const usage = resp?.response?.usage || resp?.usage;
  if (usage) {
    const inputTokens = usage.inputTokens ?? usage.prompt_tokens ?? usage.input_tokens ?? 0;
    const outputTokens = usage.outputTokens ?? usage.completion_tokens ?? usage.output_tokens ?? 0;
    const totalTokens = inputTokens + outputTokens;
    console.log(`\n[Usage] prompt=${inputTokens}, completion=${outputTokens}, total=${totalTokens}`);
  }

  // Submit the trace with all spans
  const trace = getOrCreateTrace();
  await submitTrace(trace);
}

if (require.main === module) {
  main().catch(err => {
    console.error('agentic-rag/response.tsx error:', err?.message || String(err));
    process.exit(1);
  });
}
