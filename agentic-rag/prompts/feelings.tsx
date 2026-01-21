import { getLatestDeployedPrompt, PROMPT_IDS, PROJECT_ID, extractSystemMessage, extractUserMessage } from '../fetchPayload';
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

// Default variables for Feelings prompt (EDIT YOUR QUERY HERE)
// Variable: NOTES - Runner's subjective feedback (stress levels, aches, sleep quality, motivation)
export const DEFAULT_VARIABLES = {
  NOTES: 'Runner feels extremely tired after yesterday\'s long run. Experiencing some knee discomfort but motivated to continue training.',
};

// Fetches system & user templates from the deployed Feelings prompt payload
export async function getTemplates(): Promise<{ systemTemplate: string; userTemplate: string }> {
  const deployed = await getLatestDeployedPrompt(PROMPT_IDS.FEELINGS);
  const systemTemplate = extractSystemMessage(deployed) || '';
  const userTemplate = extractUserMessage(deployed) || '';
  return { systemTemplate, userTemplate };
}

// Returns final system and user messages with variables injected (overrides optional)
export async function getInjectedMessages(overrides?: Partial<typeof DEFAULT_VARIABLES>): Promise<{
  systemMessage: string;
  userMessage: string;
}> {
    const deployedPrompt = await getLatestDeployedPrompt(PROMPT_IDS.FEELINGS);
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
    console.log('=== Feelings System Message ===');
    console.log(systemMessage);
    console.log('\n=== Feelings User Message ===');
    console.log(userMessage);
  })().catch(err => {
    console.error('feelings.tsx error:', err?.message || String(err));
    process.exit(1);
  });
}

