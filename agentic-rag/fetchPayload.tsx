import 'dotenv/config';
import { v4 as uuidv4 } from 'uuid';

const apiKey = process.env.ADALINE_API_KEY;
if (!apiKey) throw new Error('ADALINE_API_KEY missing');

// Agentic RAG deployment configuration
export const PROMPT_ID = '3b8a5264-dd69-409f-9748-7ac6dfa3772f';
const deploymentEnvironmentId = 'f73930f4-21d1-486d-a4b8-66bee70615c8';
const deploymentId = '89097614-b22f-4d24-a544-40683f750857';
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
export async function fetchDeployedPrompt(): Promise<DeployedPrompt> {
  const start = Date.now();
  
  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      }
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch prompt: ${response.status} ${response.statusText}`);
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

    console.log(`✓ Fetched deployment in ${end - start}ms`);
    return data;
  } catch (error) {
    console.error('Error fetching deployed prompt:', error);
    throw error;
  }
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
