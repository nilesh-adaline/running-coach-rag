// basics/prompt.tsx
// Defines the main user prompt(s) or templates used by the retriever.
// Supports dynamic variable injection via {{placeholder}} syntax.
// Fetches deployed prompts from Adaline and replaces placeholders with actual values.

import { Trace, addSpan } from './observability';
import { 
  fetchDeployedPrompt, 
  extractSystemMessage,
  getInjectedUserQuery,
  PROMPT_ID
} from './fetchDeployedPrompt';

// Cache for fetched prompts to avoid repeated API calls
let cachedSystemMessage: string | null = null;

/**
 * Fetch and cache the deployed system message from Adaline.
 */
async function fetchAndCacheSystemMessage(trace?: Trace): Promise<void> {
  if (cachedSystemMessage !== null) {
    return; // Already cached
  }

  try {
    const deployedPrompt = await fetchDeployedPrompt(trace);
    cachedSystemMessage = extractSystemMessage(deployedPrompt);
  } catch (error) {
    console.error('Failed to fetch deployed prompt from Adaline:', error);
    throw new Error('Unable to fetch prompt templates. Please check your API configuration.');
  }
}

/**
 * Get the coach template (system message) from deployed prompt.
 * Fetches from Adaline API on first call, then uses cache.
 * 
 * @returns The system message template
 */
export async function getCoachTemplate(trace?: Trace): Promise<string> {
  await fetchAndCacheSystemMessage(trace);
  
  if (!cachedSystemMessage) {
    throw new Error('System message template not available');
  }
  
  return cachedSystemMessage;
}

/**
 * Get the user query with variables already injected.
 * Edit DEFAULT_QUERY_VARIABLES in fetchDeployedPrompt.ts to change the query.
 * 
 * @returns The user message with variables injected
 */
export async function getUserQuery(): Promise<string> {
  return getInjectedUserQuery();
}

/**
 * Get both prompts and log to trace.
 * Used internally by retrieval and augmentation modules.
 */
export async function getFullPrompt(trace?: Trace): Promise<{ userQuery: string; coachTemplate: string }> {
  // Ensure fetch_deployed_prompt span appears BEFORE prompt_retrieval
  // by fetching and caching the system message first.
  await fetchAndCacheSystemMessage(trace);

  const startTime = Date.now();
  const userQuery = await getUserQuery();
  // Use cached system message without triggering another fetch span
  const coachTemplate = await getCoachTemplate();
  const endTime = Date.now();

  console.log('\n=== Coach Template ===');
  console.log(coachTemplate);
  console.log('\n=== User Query ===');
  console.log(userQuery);

  if (trace) {
    addSpan(trace, {
      name: 'prompt_retrieval',
      status: 'success',
      startedAt: startTime,
      endedAt: endTime,
      content: {
        type: 'Function',
        input: {
          operation: 'retrieve_and_assemble_prompt',
          fetchSystemMessage: true,
          fetchUserQuery: true,
        },
        output: {
          userQuery,
          coachTemplate,
          userQueryLength: userQuery.length,
          coachTemplateLength: coachTemplate.length,
          combinedLength: userQuery.length + coachTemplate.length,
          userQueryWords: userQuery.split(/\s+/).length,
          coachTemplateWords: coachTemplate.split(/\s+/).length,
          processedText: {
            systemMessageExtracted: true,
            variablesInjected: true,
            combinedPromptReady: true,
          },
        },
      },
      promptId: PROMPT_ID,
      latency: endTime - startTime,
      cost: 0, // No cost for prompt assembly
    });
  }

  return { userQuery, coachTemplate };
}