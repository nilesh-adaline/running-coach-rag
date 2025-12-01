import 'dotenv/config';
import { Trace, addSpan } from './observability';

const apiKey = process.env.ADALINE_API_KEY;
if (!apiKey) throw new Error('ADX_KEY missing');

// Replace with your values
export const PROMPT_ID = '3387c27c-d736-4bda-b30f-8494944d0b58';
const deploymentEnvironmentId = 'f73930f4-21d1-486d-a4b8-66bee70615c8';
export const PROJECT_ID = '843c9aa0-f1a9-4c29-b742-b8eaccd7f1a1';

const baseUrl = 'https://api.staging.adaline.ai/v2/deployments';
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
export async function fetchDeployedPrompt(trace?: Trace): Promise<DeployedPrompt> {
  const start = Date.now();
  let status: 'success' | 'error' = 'success';
  let responseData: DeployedPrompt | null = null;
  let errorMessage = '';
  let httpStatus = 0;
  
  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      }
    });

    httpStatus = response.status;

    if (!response.ok) {
      status = 'error';
      errorMessage = `Failed to fetch prompt: ${response.status} ${response.statusText}`;
      throw new Error(errorMessage);
    }

    const data = await response.json() as DeployedPrompt;
    cachedDeployedPrompt = data;
    responseData = data;
    return data;
  } finally {
    if (trace) {
      const endTime = Date.now();
      addSpan(trace, {
        name: 'fetch_deployed_payload',
        status,
        startedAt: start,
        endedAt: endTime,
        content: {
          type: 'Function',
          input: { 
            operation: 'fetch_deployed_prompt',
            url, 
            promptId: PROMPT_ID, 
            deploymentEnvironmentId,
            method: 'GET',
          },
          output: { 
            projectId: PROJECT_ID,
            httpStatus,
            promptId: responseData?.promptId,
            deploymentId: responseData?.id,
            providerName: responseData?.prompt?.config?.providerName,
            model: responseData?.prompt?.config?.model,
            messageCount: responseData?.prompt?.messages?.length || 0,
            variableCount: responseData?.prompt?.variables?.length || 0,
            toolCount: responseData?.prompt?.tools?.length || 0,
            cached: !!cachedDeployedPrompt,
            error: errorMessage || undefined,
          },
        },
        promptId: PROMPT_ID,
        latency: endTime - start,
        cost: 0, // No cost for fetching deployment config
      });
    }
  }
}

/**
 * Returns the latest deployed prompt, using cache when available.
 */
export async function getLatestDeployedPrompt(trace?: Trace): Promise<DeployedPrompt> {
  if (cachedDeployedPrompt) return cachedDeployedPrompt;
  return fetchDeployedPrompt(trace);
}

/**
 * Returns provider/model/settings/tools and identifiers from the deployment payload.
 */
export async function getDeploymentInfo(trace?: Trace): Promise<{
  providerName: string;
  model: string;
  settings: Record<string, any>;
  tools: any[];
  promptId: string;
  deploymentEnvironmentId: string;
  projectId: string;
  deploymentId: string;
}> {
  const d = await getLatestDeployedPrompt(trace);
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
  
  // Extract text from content array - handle both 'text' and 'value' properties
  const textContent = systemMsg.content.find(c => c.modality === 'text');
  return textContent?.value || textContent?.text || '';
}

/**
 * Extract the user message template from deployed prompt
 */
export function extractUserMessage(deployedPrompt: DeployedPrompt): string {
  const userMsg = deployedPrompt.prompt.messages.find(m => m.role === 'user');
  if (!userMsg) return '';
  
  // Extract text from content array - handle both 'text' and 'value' properties
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
 * 
 * @param template - The template string with {{PLACEHOLDER}} syntax
 * @param variables - Object mapping variable names to their values
 * @returns The template with all placeholders replaced
 * 
 * @example
 * const template = "Build a plan for: {{RUN_BLOCK}}.";
 * const result = injectVariables(template, { RUN_BLOCK: "5K tempo run" });
 * // Returns: "Build a plan for: 5K tempo run."
 */
export function injectVariables(template: string, variables: Record<string, any>): string {
  return template.replace(/\{\{([^}]+)\}\}/g, (match, variableName) => {
    const trimmedName = variableName.trim();
    if (trimmedName in variables) {
      return String(variables[trimmedName]);
    }
    // Keep placeholder if variable not provided
    return match;
  });
}

/**
 * Default variables for user query - EDIT THESE VALUES as needed
 */
export const DEFAULT_QUERY_VARIABLES = {
  RUN_BLOCK: '1 hour Recovery run to get rid of soreness and stiffness.',
  WHAT_TO_COVER: 'pre-workout warm-up; hydration & electrolytes; cadence tips',
  CONTEXT: 'Training for a half-marathon in 6 weeks; cool 15°C weather; access to water every 3 km; previous ankle sprain—avoid uneven terrain.'
};

/**
 * Get the user query with variables already injected.
 * Edit DEFAULT_QUERY_VARIABLES above to change the query.
 * 
 * @returns The user message with default variables injected
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
      console.log('Fetching Payload from Adaline...\n');
      
      const deployedPrompt = await fetchDeployedPrompt();
      
      console.log('✓ Payload fetched successfully');
      console.log(`  Model: ${deployedPrompt.prompt.config.model}`);
      console.log(`  Provider: ${deployedPrompt.prompt.config.providerName}\n`);
      
      const systemMessage = extractSystemMessage(deployedPrompt);
      const userMessage = extractUserMessage(deployedPrompt);
      const variables = extractVariables(deployedPrompt);
      
      console.log('\n--- System Message Template ---');
      console.log(systemMessage);
      
      console.log('\n--- User Message Template ---');
      console.log(userMessage);
      
      console.log('\n--- Variables ---');
      console.log(variables);
      
      // Example: Inject custom variables into the user message
      console.log('\n--- Example with Variable Injection ---');
      const customVariables = {
        RUN_BLOCK: "10k running in 45 minutes, with focus on pacing and breathing techniques.",
        WHAT_TO_COVER: "pre-workout warm-up; hydration & electrolytes; cadence tips",
        CONTEXT: "10K in 8 weeks; humid 30°C; access to water every 2 km; mild left-knee history—avoid deep lunges"
      };
      const injectedMessage = injectVariables(userMessage, customVariables);
      console.log(injectedMessage);
      
    } catch (error: any) {
      console.error('Error:', error.message);
      process.exit(1);
    }
  })();
}