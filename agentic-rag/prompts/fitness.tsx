import { getLatestDeployedPrompt, PROMPT_IDS, PROJECT_ID, extractSystemMessage, extractUserMessage } from '../fetchPayload';
import { DEFAULT_VARIABLES as HEAD_COACH_VARIABLES } from './headCoach';
import 'dotenv/config';

// Simple variable injector: replaces {{VAR_NAME}} in a template string
export function injectVariables(template: string, variables: Record<string, any>): string {
  return template.replace(/\{\{([^}]+)\}\}/g, (match, variableName) => {
    const key = variableName.trim();
    return Object.prototype.hasOwnProperty.call(variables, key)
      ? String(variables[key])
      : match; // leave placeholder if not provided
  });
}

// Default variables for Fitness prompt - shares same variables as Head Coach
// Variables: RUN_BLOCK, WHAT_TO_COVER, CONTEXT
// Edit queries in headCoach.tsx to update both Head Coach and Fitness prompts
export const DEFAULT_VARIABLES = HEAD_COACH_VARIABLES;

// Fetches system & user templates from the deployed Fitness prompt payload
export async function getTemplates(): Promise<{ systemTemplate: string; userTemplate: string }> {
  const deployed = await getLatestDeployedPrompt(PROMPT_IDS.FITNESS);
  const systemTemplate = extractSystemMessage(deployed) || '';
  const userTemplate = extractUserMessage(deployed) || '';
  return { systemTemplate, userTemplate };
}

// Returns final system and user messages with variables injected (overrides optional)
export async function getInjectedMessages(overrides?: Partial<typeof DEFAULT_VARIABLES>): Promise<{
  systemMessage: string;
  userMessage: string;
}> {
    const deployedPrompt = await getLatestDeployedPrompt(PROMPT_IDS.FITNESS);
    const { systemTemplate, userTemplate } = await getTemplates();
    const vars = { ...DEFAULT_VARIABLES, ...(overrides || {}) };
    const systemMessage = injectVariables(systemTemplate, vars);
    const userMessage = injectVariables(userTemplate, vars);

    // Observability handled by response.tsx
    return { systemMessage, userMessage };
}

// CLI demo
if (require.main === module) {
  (async () => {
    const { systemMessage, userMessage } = await getInjectedMessages();
    console.log('=== Fitness System Message ===');
    console.log(systemMessage);
    console.log('\n=== Fitness User Message ===');
    console.log(userMessage);
  })().catch(err => {
    console.error('fitness.tsx error:', err?.message || String(err));
    process.exit(1);
  });
}

