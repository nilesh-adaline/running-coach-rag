import 'dotenv/config';
import { v4 as uuidv4 } from 'uuid';

const apiKey = process.env.ADALINE_API_KEY;
if (!apiKey) {
  console.error('❌ ADALINE_API_KEY is not set in environment variables.');
  console.error('   Please set it in your .env file or export it: export ADALINE_API_KEY=your_key');
  throw new Error('ADALINE_API_KEY missing');
}

// Deployment configuration
export const PROJECT_ID = '843c9aa0-f1a9-4c29-b742-b8eaccd7f1a1';
const deploymentEnvironmentId = 'f73930f4-21d1-486d-a4b8-66bee70615c8';
const baseUrl = 'https://api.staging.adaline.ai/v2/deployments';

// Prompt configurations
export const PROMPT_IDS = {
  AGENTIC_RAG: '3b8a5264-dd69-409f-9748-7ac6dfa3772f',
  FEELINGS: '10a3201f-5716-48dd-a62e-28356560aed9',
  FITNESS: '2c7e4ac7-901e-4fe4-8ba4-e1ba7cd7228f',
} as const;

// For backward compatibility
export const PROMPT_ID = PROMPT_IDS.AGENTIC_RAG;

// Type definitions for the deployed prompt structure
export interface DeployedPromptMessage {
  role: 'system' | 'user' | 'assistant';
  content: Array<{ modality: string; text?: string; value?: string }>;
}

export interface DeployedPrompt {
  id: string;
  promptId: string;
  projectId: string;
  deploymentEnvironmentId: string;
  prompt: {
    config: {
      providerName: string;
      model: string;
      settings: Record<string, any>;
    };
    messages: DeployedPromptMessage[];
    tools: any[];
    variables: Array<{ name: string; description?: string }>;
  };
}

// Cache for deployed prompts (keyed by promptId)
const promptCache: Map<string, DeployedPrompt> = new Map();

/**
 * Generic function to fetch a deployed prompt from Adaline API
 * @param promptId - The prompt ID to fetch
 * @param deploymentEnvironmentId - The deployment environment ID (defaults to shared env)
 */
export async function fetchDeployedPrompt(
  promptId: string = PROMPT_IDS.AGENTIC_RAG,
  deploymentEnvironmentId: string = 'f73930f4-21d1-486d-a4b8-66bee70615c8'
): Promise<DeployedPrompt> {
  const url = `${baseUrl}?promptId=${promptId}&deploymentEnvironmentId=${deploymentEnvironmentId}&deploymentId=latest`;
  const start = Date.now();
  
  // Retry logic for network errors - increased retries and better handling
  const maxRetries = 4;
  const baseRetryDelay = 1000; // Start with 1 second
  let lastError: any = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      // Add a small delay before retries to let connections stabilize (except first attempt)
      if (attempt > 0) {
        const delay = baseRetryDelay * Math.pow(2, attempt - 1);
        const cappedDelay = Math.min(delay, 8000); // Cap at 8 seconds
        await new Promise(resolve => setTimeout(resolve, cappedDelay));
      }
      
      // Add timeout to fetch request
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 45000); // Increased to 45 second timeout
      
      try {
        const response = await fetch(url, {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'Adaline-Agentic-RAG/1.0'
          },
          signal: controller.signal,
          // Remove keep-alive header as it may cause issues with Node.js fetch
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const errorText = await response.text().catch(() => 'No error details');
          const errorMsg = `Failed to fetch prompt: ${response.status} ${response.statusText}. ${errorText}`;
          if (attempt < maxRetries && (response.status >= 500 || response.status === 429)) {
            // Retry on server errors and rate limits
            const delay = baseRetryDelay * Math.pow(2, attempt);
            console.warn(`${errorMsg} (Attempt ${attempt + 1}/${maxRetries + 1}, retrying in ${delay}ms...)`);
            await new Promise(resolve => setTimeout(resolve, delay));
            continue;
          }
          throw new Error(errorMsg);
        }

        const data = await response.json() as DeployedPrompt;
        promptCache.set(promptId, data);
        const end = Date.now();

        // Span will be added by response.tsx after import
        (global as any).__fetchPayloadSpan = {
          name: 'fetch_payload',
          start,
          end,
          data: {
            providerName: data.prompt.config.providerName,
            model: data.prompt.config.model,
            toolsCount: (data.prompt.tools || []).length,
            variableNames: (data.prompt.variables || []).map(v => v.name),
          },
        };

        // Only show success message if it took more than 1 attempt
        if (attempt > 0) {
          console.log(`✓ Fetched deployment in ${end - start}ms (succeeded after ${attempt + 1} attempts)`);
        } else {
          console.log(`✓ Fetched deployment in ${end - start}ms`);
        }
        return data;
      } catch (fetchError: any) {
        // Re-throw fetch errors to outer catch block
        throw fetchError;
      }
    } catch (error: any) {
      lastError = error;
      
      // Check for network errors in error or cause - expanded detection
      const errorCode = error.code || error.cause?.code;
      const errorMessage = error.message || error.cause?.message || String(error);
      const errorName = error.name || error.cause?.name || '';
      
      const isNetworkError = 
        errorCode === 'ECONNRESET' || 
        errorCode === 'ECONNREFUSED' || 
        errorCode === 'ENOTFOUND' ||
        errorCode === 'ETIMEDOUT' ||
        errorCode === 'ECONNABORTED' ||
        errorName === 'AbortError' ||
        errorName === 'TimeoutError' ||
        errorMessage.includes('fetch failed') ||
        errorMessage.includes('network') ||
        errorMessage.includes('ECONNRESET') ||
        errorMessage.includes('socket') ||
        errorMessage.includes('connection');
      
      if (isNetworkError && attempt < maxRetries) {
        const nextDelay = baseRetryDelay * Math.pow(2, attempt); // Delay for NEXT attempt
        const errorInfo = errorCode || errorName || 'network error';
        console.warn(`Network error (${errorInfo}): Retrying in ${nextDelay}ms... (Attempt ${attempt + 1}/${maxRetries + 1})`);
        // Delay is handled at the start of the next loop iteration
        continue;
      }
      
      // Last attempt failed or non-network error
      if (errorName === 'AbortError' || errorMessage.includes('timeout')) {
        const err = new Error('Request timeout: Failed to fetch deployed prompt within 45 seconds. Check your network connection and API endpoint availability.');
        console.error('Error fetching deployed prompt:', err.message);
        throw err;
      }
      
      if (isNetworkError) {
        const err = new Error(`Network error: Cannot connect to Adaline API at ${baseUrl}.\n` +
          `Error: ${errorCode || errorName || 'unknown'}\n` +
          `Attempted ${attempt + 1} times with exponential backoff.\n` +
          `Please check:\n` +
          `  1. Network connectivity to ${baseUrl}\n` +
          `  2. ADALINE_API_KEY is set correctly\n` +
          `  3. API endpoint is accessible\n` +
          `  4. Firewall/proxy settings allow outbound connections`);
        console.error('Error fetching deployed prompt:', err.message);
        throw err;
      }
      
      // Non-network errors (e.g., JSON parsing, auth errors)
      if (error.message) {
        console.error('Error fetching deployed prompt:', error.message);
      } else {
        console.error('Error fetching deployed prompt:', error);
      }
      throw error;
    }
  }
  
  // Should never reach here, but TypeScript needs it
  throw lastError || new Error('Failed to fetch deployed prompt after retries');
}

/**
 * Returns the latest deployed prompt, using cache when available.
 * @param promptId - The prompt ID to fetch (defaults to AGENTIC_RAG)
 */
export async function getLatestDeployedPrompt(promptId: string = PROMPT_IDS.AGENTIC_RAG): Promise<DeployedPrompt> {
  if (promptCache.has(promptId)) return promptCache.get(promptId)!;
  return fetchDeployedPrompt(promptId);
}

/**
 * Fetch the Feelings prompt deployment
 */
export async function fetchFeelingsPrompt(): Promise<DeployedPrompt> {
  return fetchDeployedPrompt(PROMPT_IDS.FEELINGS);
}

/**
 * Fetch the Fitness prompt deployment
 */
export async function fetchFitnessPrompt(): Promise<DeployedPrompt> {
  return fetchDeployedPrompt(PROMPT_IDS.FITNESS);
}

/**
 * Returns provider/model/settings/tools and identifiers from the deployment payload.
 */
export async function getDeploymentInfo(): Promise<{
  providerName: string;
  model: string;
  settings: Record<string, any>;
  tools: any[];
  promptId: string;
  deploymentEnvironmentId: string;
  projectId: string;
  deploymentId: string;
}> {
  const d = await getLatestDeployedPrompt();
  return {
    providerName: d.prompt.config.providerName,
    model: d.prompt.config.model,
    settings: d.prompt.config.settings || {},
    tools: d.prompt.tools || [],
    promptId: d.promptId,
    deploymentEnvironmentId: d.deploymentEnvironmentId,
    projectId: d.projectId,
    deploymentId: d.id,
  };
}

/**
 * Extract the system message template from deployed prompt
 */
export function extractSystemMessage(deployedPrompt: DeployedPrompt): string {
  const systemMsg = deployedPrompt.prompt.messages.find(m => m.role === 'system');
  if (!systemMsg) return '';
  
  const textContent = systemMsg.content.find(c => c.modality === 'text');
  return textContent?.value || textContent?.text || '';
}

/**
 * Extract the user message template from deployed prompt
 */
export function extractUserMessage(deployedPrompt: DeployedPrompt): string {
  const userMsg = deployedPrompt.prompt.messages.find(m => m.role === 'user');
  if (!userMsg) return '';
  
  const textContent = userMsg.content.find(c => c.modality === 'text');
  return textContent?.value || textContent?.text || '';
}

/**
 * Extract variable names from deployed prompt
 */
export function extractVariables(deployedPrompt: DeployedPrompt): string[] {
  return deployedPrompt.prompt.variables.map(v => v.name);
}

/**
 * Inject user-defined variables into a template string.
 * Replaces {{VARIABLE_NAME}} placeholders with actual values.
 */
export function injectVariables(template: string, variables: Record<string, any>): string {
  return template.replace(/\{\{([^}]+)\}\}/g, (match, variableName) => {
    const trimmedName = variableName.trim();
    if (trimmedName in variables) {
      return String(variables[trimmedName]);
    }
    return match;
  });
}

/**
 * Default variables for Head Coach prompt
 * Variables: RUN_BLOCK, WHAT_TO_COVER, CONTEXT, WORKOUT_SPLITS, ADVICE
 */
export const DEFAULT_AGENTIC_RAG_VARIABLES = {
  RUN_BLOCK: 'Base Building Phase',
  WHAT_TO_COVER: 'Training principles and recovery strategies',
  CONTEXT: 'User is a beginner runner preparing for their first 5K',
  WORKOUT_SPLITS: '3 runs per week',
  ADVICE: 'Focus on consistency over intensity',
};

/**
 * Default variables for Feelings prompt
 * Variables: NOTES
 */
export const DEFAULT_FEELINGS_VARIABLES = {
  NOTES: 'Runner feels tired after yesterday\'s long run. Experiencing some knee discomfort but motivated to continue training.',
};

/**
 * Default variables for Fitness prompt
 * Variables: RUN_BLOCK, WHAT_TO_COVER
 */
export const DEFAULT_FITNESS_VARIABLES = {
  RUN_BLOCK: 'Base Building Phase',
  WHAT_TO_COVER: 'Strength training and mobility exercises',
};

// For backward compatibility
export const DEFAULT_QUERY_VARIABLES = DEFAULT_AGENTIC_RAG_VARIABLES;

/**
 * Get the user query with variables already injected.
 */
export async function getInjectedUserQuery(): Promise<string> {
  const deployedPrompt = await getLatestDeployedPrompt();
  const userMessageTemplate = extractUserMessage(deployedPrompt);
  return injectVariables(userMessageTemplate, DEFAULT_QUERY_VARIABLES);
}

/**
 * Helper function to display prompt details
 */
function displayPromptDetails(name: string, deployedPrompt: DeployedPrompt) {
  console.log(`\n=== ${name} Prompt ===`);
  console.log(`  Model: ${deployedPrompt.prompt.config.model}`);
  console.log(`  Provider: ${deployedPrompt.prompt.config.providerName}`);
  console.log(`  Deployment ID: ${deployedPrompt.id}`);
  console.log(`  Prompt ID: ${deployedPrompt.promptId}`);
  console.log(`  Tools count: ${(deployedPrompt.prompt.tools || []).length}`);
  
  const systemMessage = extractSystemMessage(deployedPrompt);
  const userMessage = extractUserMessage(deployedPrompt);
  const variables = extractVariables(deployedPrompt);
  
  console.log('\n  System Message:');
  console.log(`  ${systemMessage.substring(0, 150)}...`);
  
  console.log('\n  User Message Template:');
  console.log(`  ${userMessage.substring(0, 150)}...`);
  
  console.log('\n  Variables:');
  variables.forEach(v => console.log(`    - ${v}`));
  
  if (deployedPrompt.prompt.tools && deployedPrompt.prompt.tools.length > 0) {
    console.log('\n  Tools:');
    deployedPrompt.prompt.tools.forEach((tool, idx) => {
      console.log(`    ${idx + 1}. ${tool.function?.name || tool.name || 'unnamed'}`);
    });
  }
}

// CLI runner - fetch and display all prompts
if (require.main === module) {
  (async () => {
    try {
      console.log('Fetching all prompts from Adaline...\n');
      
      // Fetch Head Coach prompt
      console.log('Fetching Head Coach prompt...');
      const agenticRagPrompt = await fetchDeployedPrompt(PROMPT_IDS.AGENTIC_RAG);
      displayPromptDetails('Head Coach', agenticRagPrompt);
      
      // Fetch Feelings prompt
      console.log('\n\nFetching Feelings prompt...');
      const feelingsPrompt = await fetchFeelingsPrompt();
      displayPromptDetails('Feelings', feelingsPrompt);
      
      // Fetch Fitness prompt
      console.log('\n\nFetching Fitness prompt...');
      const fitnessPrompt = await fetchFitnessPrompt();
      displayPromptDetails('Fitness', fitnessPrompt);
      
      console.log('\n\n✓ All prompts fetched successfully!');
      
    } catch (error: any) {
      console.error('Error:', error.message);
      process.exit(1);
    }
  })();
}


