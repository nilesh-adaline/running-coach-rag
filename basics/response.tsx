// basics/response.tsx
// This file orchestrates the entire RAG pipeline with observability.
// 1) Creates a trace for the entire operation.
// 2) Assembles an augmented prompt, creating spans for retrieval and prompt construction.
// 3) Calls the LLM with the augmented prompt, creating a span for the model call.
// 4) Prints the final response.
// 5) Submits the complete trace to the Adaline API.

import 'dotenv/config';
import { OpenAI } from '@adaline/openai';
import { Gateway } from '@adaline/gateway';
import { Config, MessageType } from '@adaline/types';
import { retrieveTopK, readChunkContent, parseMatchMetadata } from './retrieve';
import { buildAugmentedPrompt } from './augmentPrompt';
import { getFullPrompt } from './prompt';
import { getDeploymentInfo, PROMPT_ID, DEFAULT_QUERY_VARIABLES } from './fetchDeployedPrompt';
import { 
  createTrace, 
  submitTrace, 
  addSpan, 
  ModelSpanContent 
} from './observability';
import type { Trace } from './observability';

const openai = new OpenAI();
const gateway = new Gateway();

/**
 * Assembles an augmented prompt by retrieving context and building the final prompt structure.
 * This function creates a parent "Function" span that wraps the inner retrieval and prompt building spans.
 * Returns separate system and user messages for clean dashboard display.
 * @param trace The main trace object.
 * @returns Object containing systemMessage and userMessage with retrieved context.
 */
async function assembleAugmentedPrompt(
  trace: Trace, 
  coachTemplate: string, 
  userQuery: string
): Promise<{ systemMessage: string; userMessage: string }> {
  const parentStartTime = Date.now();
  const parentReferenceId = `assemble_augmented_prompt_${Date.now()}`;
  
  const snippets: string[] = [];
  console.log('Retrieving topK matches from vector DB...');
  const fullPrompt = `${coachTemplate}\n\nUser request:\n${userQuery}`;

  // Ensure prompt spans are recorded before embeddings/retrieval by passing override
  const matches = await retrieveTopK(5, trace, fullPrompt, parentReferenceId);
  console.log(`Retrieved ${matches.length} match(es)`);

  for (let i = 0; i < matches.length; i++) {
    const m = matches[i];
    const { fileName, chunkNum } = await parseMatchMetadata(m);
    if (fileName && typeof chunkNum === 'number') {
      const content = await readChunkContent(fileName, chunkNum);
      if (content && content.trim()) {
        snippets.push(content);
      }
    }
  }

  // Build system message with coach template and retrieved context
  // Capture augmentation start AFTER retrieval has finished so span ordering is correct.
  const augmentationStartTime = Date.now();
  let systemMessage = coachTemplate;
  if (snippets.length > 0) {
    systemMessage += `\n\nContext from knowledge base (use only when relevant):\n`;
    for (let i = 0; i < snippets.length; i++) {
      systemMessage += `--- snippet ${i + 1} ---\n${snippets[i]}\n\n`;
    }
  }
  
  // Build user message with the query and instructions
  const userMessage = `${userQuery}\n\nInstructions: Use ONLY the requested sections from the template. Keep answers tight and metric. If pain or heat is a concern, add a short caution.`;
  
  console.log(`Augmented prompt assembled: system=${systemMessage.length} chars, user=${userMessage.length} chars`);

  // Add dedicated prompt_augmentation span to capture detailed prompt construction metrics
  const augmentationEndTime = Date.now();
  const contextBlock = snippets.join('\n\n');
  const coachWordCount = coachTemplate.split(/\s+/).length;
  const userQueryWordCount = userQuery.split(/\s+/).length;
  const contextWordCount = contextBlock ? contextBlock.split(/\s+/).length : 0;
  const totalWordCount = (systemMessage + '\n' + userMessage).split(/\s+/).length;
  addSpan(trace, {
    name: 'prompt_augmentation',
    status: 'success',
    startedAt: augmentationStartTime, // start AFTER retrieval for proper ordering
    endedAt: augmentationEndTime,
    parentReferenceId: parentReferenceId,
    content: {
      type: 'Function',
      input: {
        operation: 'augment_prompt_with_retrieval_context',
        snippetsIncluded: snippets.length,
        coachTemplateLength: coachTemplate.length,
        userQueryLength: userQuery.length,
        contextSnippetsLengths: snippets.map(s => s.length),
        totalContextLength: snippets.reduce((sum, s) => sum + s.length, 0),
      },
      output: {
        systemMessageLength: systemMessage.length,
        userMessageLength: userMessage.length,
        components: ['coach_template', 'user_query', 'retrieval_context', 'instructions'],
        componentLengths: {
          coachTemplate: coachTemplate.length,
          userQuery: userQuery.length,
          retrievalContext: contextBlock.length,
        },
        wordCounts: {
          coachTemplate: coachWordCount,
          userQuery: userQueryWordCount,
          retrievalContext: contextWordCount,
          total: totalWordCount,
        },
        snippetsProcessed: snippets.length,
        estimatedTokens: Math.ceil((systemMessage.length + userMessage.length) / 4),
      },
    },
    promptId: PROMPT_ID,
  latency: augmentationEndTime - augmentationStartTime,
    cost: 0,
  });
  
  // Add parent span that wraps retrieval + augmentation
  const parentEndTime = Date.now();
  addSpan(trace, {
    name: 'assemble_augmented_prompt',
    status: 'success',
    startedAt: parentStartTime,
    endedAt: parentEndTime,
    referenceId: parentReferenceId,
    content: {
      type: 'Function',
      input: {
        operation: 'assemble_augmented_prompt_pipeline',
        coachTemplateLength: coachTemplate.length,
        userQueryLength: userQuery.length,
        topK: 5,
      },
      output: {
        systemMessageLength: systemMessage.length,
        userMessageLength: userMessage.length,
        snippetsRetrieved: snippets.length,
        matchesFound: matches.length,
        pipelineSteps: ['embedding_create', 'pinecone_query', 'prompt_augmentation'],
      },
    },
    promptId: PROMPT_ID,
    latency: parentEndTime - parentStartTime,
    cost: 0,
  });
  
  return { systemMessage, userMessage };
}

/**
 * Calls the LLM via Adaline Gateway with system and user messages.
 * Creates a "Model" span to record the interaction.
 * @param systemMessage The system message (coach template with context).
 * @param userMessage The user message (user query with context).
 * @param trace The main trace object.
 * @returns The assistant's response text.
 */
async function callLLM(systemMessage: string, userMessage: string, trace: Trace): Promise<string> {
  const startTime = Date.now();
  const apiKey = process.env.OAI_API_KEY;
  if (!apiKey) throw new Error('OAI_API_KEY missing');

  // Pull dynamic model/config from deployment payload
  const info = await getDeploymentInfo(trace);
  const providerName = info.providerName;
  const modelName = info.model;
  const config = Config().parse(info.settings);
  const model = openai.chatModel({ modelName, apiKey });

  const messages: MessageType[] = [
    { role: 'system', content: [{ modality: 'text', value: systemMessage }] },
    { role: 'user', content: [{ modality: 'text', value: userMessage }] },
  ];

  let resp: any;
  let status: 'success' | 'error' = 'success';
  let errorMsg = '';
  let outputMessage: any = {};
  let inputTokens = 0;
  let outputTokens = 0;
  let finishReason = '';
  let responseId = '';
  let completionDetails: any = {};
  // Default request config (may be overridden for specific model compatibility)
  let requestConfig = config;

  try {
    // Some newer models (e.g. gpt-5-mini) only accept the default temperature (1).
    // Ensure we pass a request config compatible with the selected deployment.
    const requestConfig = { ...config, temperature: /gpt-5/.test(String(modelName)) ? 1 : config.temperature };

    console.log(`Calling LLM (${modelName}) via Adaline Gateway...`);
    resp = await gateway.completeChat({ model, config: requestConfig, messages });
    
    outputMessage = resp?.response?.messages?.[0] ?? {};
    finishReason = resp?.response?.finishReason || resp?.response?.finish_reason || 'unknown';
    responseId = resp?.response?.id || resp?.id || '';
    
    // Extract full completion details for the span (similar to OpenAI response format)
    // Build usage object from response
    const usage = resp?.response?.usage || resp?.usage;
    let usageObj: any = null;
    const hasUsage = !!(
      usage && (
        usage.inputTokens !== undefined ||
        usage.prompt_tokens !== undefined ||
        usage.input_tokens !== undefined
      )
    );
    
    if (hasUsage) {
      usageObj = {
        prompt_tokens: usage.inputTokens ?? usage.prompt_tokens ?? usage.input_tokens ?? 0,
        completion_tokens: usage.outputTokens ?? usage.completion_tokens ?? usage.output_tokens ?? 0,
        total_tokens: (usage.inputTokens ?? usage.prompt_tokens ?? usage.input_tokens ?? 0) +
                      (usage.outputTokens ?? usage.completion_tokens ?? usage.output_tokens ?? 0),
      };
      inputTokens = usageObj.prompt_tokens;
      outputTokens = usageObj.completion_tokens;
    }
    
    completionDetails = {
      id: responseId || `chatcmpl-${Date.now()}`,
      object: resp?.response?.object || 'chat.completion',
      created: resp?.response?.created || Math.floor(Date.now() / 1000),
      model: resp?.response?.model || modelName,
      system_fingerprint: resp?.response?.system_fingerprint || null,
      choices: resp?.response?.choices || [{
        index: 0,
        message: {
          role: outputMessage?.role || 'assistant',
          content: outputMessage?.content?.find((c: any) => c.modality === 'text')?.value || '',
          refusal: null,
        },
        logprobs: null,
        finish_reason: finishReason,
      }],
      ...(usageObj ? { usage: usageObj } : {}),
    };
  } catch (err: any) {
    status = 'error';
    errorMsg = err instanceof Error ? err.message : String(err);
    console.error("Error calling LLM:", errorMsg);
    throw err; // Re-throw to be caught by the main pipeline
  } finally {
    const endTime = Date.now();
    
  // Extract output text for analysis (may be undefined if provider did not return text)
  const outputText = completionDetails?.choices?.[0]?.message?.content || '';

  // GPT-4.1-mini pricing: $0.30 per 1M input tokens, $1.20 per 1M output tokens
  const inputCostPer1M = 0.30;
  const outputCostPer1M = 1.20;
  // For cost, fall back to a conservative token estimate if provider didn't return usage
  const combinedInputLength = systemMessage.length + userMessage.length;
  const inputTokensForCost = inputTokens || Math.ceil(combinedInputLength / 4);
  const outputTokensForCost = outputTokens || Math.ceil(outputText.length / 4);
  const llmCost = (inputTokensForCost / 1_000_000) * inputCostPer1M + (outputTokensForCost / 1_000_000) * outputCostPer1M;
    
  console.log(`   💰 LLM cost: $${llmCost.toFixed(6)} (${inputTokensForCost} input + ${outputTokensForCost} output = ${inputTokensForCost + outputTokensForCost} total tokens)`);

    // Prepare output using Adaline MessageType format
    // Output should contain messages array with assistant role
    const outputPayload = {
      messages: [{
        role: 'assistant',
        content: [{
          modality: 'text',
          value: outputText,
        }],
      }],
      ...(inputTokens || outputTokens ? {
        tokenUsage: {
          promptTokens: inputTokens,
          completionTokens: outputTokens,
          totalTokens: inputTokens + outputTokens,
        }
      } : {}),
    };

    // Convert variables to the format expected by Adaline API
    // Each variable value should be of ContentType format with modality and value
    const variables: Record<string, { modality: string; value: string }> = {};
    for (const [key, val] of Object.entries(DEFAULT_QUERY_VARIABLES)) {
      variables[key] = {
        modality: 'text',
        value: String(val),
      };
    }
    
    addSpan(trace, {
      name: 'llm_call',
      status,
      startedAt: startTime,
      endedAt: endTime,
      content: {
        type: 'Model',
        provider: providerName,
        model: modelName,
        variables, // Add variables at content level
        input: {
          model: modelName,
          max_tokens: requestConfig.max_output_tokens || requestConfig.maxTokens,
          temperature: requestConfig.temperature,
          messages: [
            {
              role: 'system',
              content: [{
                type: 'text',
                text: systemMessage,
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
        },
        output: outputPayload,
      } as any,
      promptId: PROMPT_ID,
      deploymentId: info.deploymentId,
      runEvaluation: true, // Enable "Add to Dataset" toggle
      latency: endTime - startTime,
      cost: llmCost,
      ...(inputTokens || outputTokens ? {
        tokens: {
          input: inputTokens,
          output: outputTokens,
          total: inputTokens + outputTokens,
        }
      } : {}),
    });
  }

  // Extract text from the response
  const assistantMsg = resp?.response?.messages?.[0];
  const textContent = assistantMsg?.content?.find((c: any) => c.modality === 'text' && c.value);
  return textContent?.value ?? '';
}

/**
 * Main execution function.
 */
async function main() {
  // 1. Initialize the trace
  const trace = createTrace('The Running Coach App (RAG)');
  
  try {
    // 2. Fetch prompt first to enforce ordering: fetch_deployed_prompt -> prompt_retrieval
    const { coachTemplate, userQuery } = await getFullPrompt(trace);
    const { systemMessage, userMessage } = await assembleAugmentedPrompt(trace, coachTemplate, userQuery);
    
    // 3. Call the LLM with separate system and user messages
    const assistantResponse = await callLLM(systemMessage, userMessage, trace);

    // 4. Print the final result
    console.log('\n================================\n');
    console.log('✅ Assistant response:\n');
    console.log(assistantResponse);
    console.log('\n================================\n');
    
    // Calculate total cost and breakdown from all spans
    const totalCost = trace.spans.reduce((sum, span) => sum + (span.cost || 0), 0);
    const totalLatency = trace.spans.reduce((sum, span) => sum + (span.latency || 0), 0);
    const embeddingCost = trace.spans.find(s => s.name === 'embedding_create')?.cost || 0;
    const llmCost = trace.spans.find(s => s.name === 'llm_call')?.cost || 0;
    
    console.log('📊 Pipeline Metrics:');
    console.log(`   Embedding Cost:  $${embeddingCost.toFixed(6)}`);
    console.log(`   LLM Cost:        $${llmCost.toFixed(6)}`);
    console.log(`   ─────────────────────────────`);
    console.log(`   Total Cost:      $${totalCost.toFixed(6)}`);
    console.log(`   Total Latency:   ${totalLatency}ms`);
    console.log(`   Spans Executed:  ${trace.spans.length}`);
    console.log('');

  } catch (error) {
    console.error('Pipeline failed:', error);
    // Mark the entire trace as failed
    trace.status = 'error';
  } finally {
    // 5. Finalize and submit the trace, regardless of success or failure
    console.log('Submitting trace to Adaline...');
    await submitTrace(trace);
  }
}

main().catch(e => {
  console.error('Unhandled error in main execution:', e);
  process.exit(1);
});
