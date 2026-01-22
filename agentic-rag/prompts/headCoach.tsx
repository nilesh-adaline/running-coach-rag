import { getLatestDeployedPrompt, PROMPT_ID, PROJECT_ID, extractSystemMessage, extractUserMessage } from '../fetchPayload';
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

// Default variables (can be overridden when calling helpers below)
export const DEFAULT_VARIABLES = {
  RUN_BLOCK: '8km tempo run this Thursday at 5:00 PM in Austin, Texas',
  WHAT_TO_COVER: 're-run Nutrition, During-run Fueling & Hydration, Post-run Recovery Meal',
  CONTEXT: 'Help me with Half marathon prep, tempo pace target is 4:30/km. Running after work, last meal at noon (office lunch). Hot climate, usually 28-32°C in afternoons. I sweat heavily. Advanced runner but new to structured fueling. Vegan diet. No injuries. Need practical foods I can prep quickly or buy near my office.'
};

// Fetches system & user templates from the deployed payload
export async function getTemplates(): Promise<{ systemTemplate: string; userTemplate: string }> {
  const deployed = await getLatestDeployedPrompt();
  const systemTemplate = extractSystemMessage(deployed) || '';
  const userTemplate = extractUserMessage(deployed) || '';
  return { systemTemplate, userTemplate };
}

// Returns final system and user messages with variables injected (overrides optional)
export async function getInjectedMessages(overrides?: Partial<typeof DEFAULT_VARIABLES>): Promise<{
  systemMessage: string;
  userMessage: string;
}> {
    const deployedPrompt = await getLatestDeployedPrompt();
    const { systemTemplate, userTemplate } = await getTemplates();
    const vars = { ...DEFAULT_VARIABLES, ...(overrides || {}) };
    const systemMessage = injectVariables(systemTemplate, vars);
    const userMessage = injectVariables(userTemplate, vars);

    // Observability handled by response.tsx
    return { systemMessage, userMessage };
}// CLI demo
if (require.main === module) {
  (async () => {
    const { systemMessage, userMessage } = await getInjectedMessages();
    console.log('=== System Message ===');
    console.log(systemMessage);
    console.log('\n=== User Message ===');
    console.log(userMessage);
  })().catch(err => {
    console.error('headCoach.tsx error:', err?.message || String(err));
    process.exit(1);
  });
}

