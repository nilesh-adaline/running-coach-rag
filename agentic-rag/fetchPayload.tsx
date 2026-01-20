import 'dotenv/config';
import { v4 as uuidv4 } from 'uuid';

const apiKey = process.env.ADALINE_API_KEY;
if (!apiKey) {
  console.error('❌ ADALINE_API_KEY is not set in environment variables.');
  console.error('   Please set it in your .env file or export it: export ADALINE_API_KEY=your_key');
  throw new Error('ADALINE_API_KEY missing');
}

// Agentic RAG deployment configuration
export const PROMPT_ID = '3b8a5264-dd69-409f-9748-7ac6dfa3772f';
const deploymentEnvironmentId = 'f73930f4-21d1-486d-a4b8-66bee70615c8';
const deploymentId = '09c78ec5-4fa3-4200-8f71-68bd153e4c8f';
export const PROJECT_ID = '843c9aa0-f1a9-4c29-b742-b8eaccd7f1a1';

const baseUrl = 'https://api.staging.adaline.ai/v2/deployments';
// Construct the full URL with required query parameters: promptId, deploymentEnvironmentId, and deploymentId=latest
const url = `${baseUrl}?promptId=${PROMPT_ID}&deploymentEnvironmentId=${deploymentEnvironmentId}&deploymentId=latest`;

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

// Cache the latest deployed prompt to avoid repeated network calls
let cachedDeployedPrompt: DeployedPrompt | null = null;

/**
 * Fetch the deployed prompt from Adaline API
 */
export async function fetchDeployedPrompt(): Promise<DeployedPrompt> {
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
        cachedDeployedPrompt = data;
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
 */
export async function getLatestDeployedPrompt(): Promise<DeployedPrompt> {
  if (cachedDeployedPrompt) return cachedDeployedPrompt;
  return fetchDeployedPrompt();
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
 * Default variables for agentic RAG query
 */
export const DEFAULT_QUERY_VARIABLES = {
  USER_QUERY: 'What are the best practices for recovery runs?',
};

/**
 * Get the user query with variables already injected.
 */
export async function getInjectedUserQuery(): Promise<string> {
  const deployedPrompt = await getLatestDeployedPrompt();
  const userMessageTemplate = extractUserMessage(deployedPrompt);
  return injectVariables(userMessageTemplate, DEFAULT_QUERY_VARIABLES);
}

// CLI runner - fetch and display the deployed prompt
if (require.main === module) {
  (async () => {
    try {
      console.log('Fetching Agentic RAG Payload from Adaline...\n');
      
      const deployedPrompt = await fetchDeployedPrompt();
      
      console.log('✓ Payload fetched successfully');
      console.log(`  Model: ${deployedPrompt.prompt.config.model}`);
      console.log(`  Provider: ${deployedPrompt.prompt.config.providerName}`);
      console.log(`  Deployment ID: ${deploymentId}`);
      console.log(`  Tools count: ${(deployedPrompt.prompt.tools || []).length}\n`);
      
      const systemMessage = extractSystemMessage(deployedPrompt);
      const userMessage = extractUserMessage(deployedPrompt);
      const variables = extractVariables(deployedPrompt);
      
      console.log('=== System Message Template ===');
      console.log(systemMessage);
      
      console.log('\n=== User Message Template ===');
      console.log(userMessage);
      
      console.log('\n=== Variables ===');
      console.log(variables);
      
      console.log('\n=== Tools ===');
      const tools = deployedPrompt.prompt.tools || [];
      tools.forEach((tool, idx) => {
        console.log(`${idx + 1}. ${tool.function?.name || tool.name || 'unnamed'}`);
        if (tool.function?.description) {
          console.log(`   Description: ${tool.function.description.substring(0, 100)}...`);
        }
      });
      
      console.log('\n=== Example with Variable Injection ===');
      const customVariables = {
        USER_QUERY: "How should I structure my training for a marathon in 12 weeks?"
      };
      const injectedMessage = injectVariables(userMessage, customVariables);
      console.log(injectedMessage);
      
    } catch (error: any) {
      console.error('Error:', error.message);
      process.exit(1);
    }
  })();
}


