"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
// Quick test to verify span structure
require("dotenv/config");
const observability_1 = require("./observability");
const fetchDeployedPrompt_1 = require("./fetchDeployedPrompt");
const trace = (0, observability_1.createTrace)('Test Trace');
// Add a mock LLM span
(0, observability_1.addSpan)(trace, {
    name: 'llm_call',
    status: 'success',
    startedAt: Date.now(),
    endedAt: Date.now() + 1000,
    content: {
        type: 'Model',
        provider: 'openai',
        model: 'gpt-4.1-mini',
        variables: {
            TEST_VAR: {
                modality: 'text',
                value: 'test value',
            },
        },
        input: {
            model: 'gpt-4.1-mini',
            max_tokens: 500,
            temperature: 0.7,
            messages: [
                {
                    role: 'system',
                    content: [{
                            type: 'text',
                            text: 'You are a helpful assistant.',
                        }],
                },
                {
                    role: 'user',
                    content: [{
                            type: 'text',
                            text: 'Hello!',
                        }],
                },
            ],
        },
        output: {
            messages: [{
                    role: 'assistant',
                    content: [{
                            modality: 'text',
                            value: 'Hello! How can I help you?',
                        }],
                }],
        },
    },
    promptId: fetchDeployedPrompt_1.PROMPT_ID,
    latency: 1000,
    cost: 0.001,
    tokens: {
        input: 50,
        output: 20,
        total: 70,
    },
});
// Map the span as the submitTrace function does
const s = trace.spans[0];
const c = s.content ?? {};
const type = c.type;
let content = {};
if (type) {
    content.type = type;
}
if (c.input !== undefined) {
    content.input = typeof c.input === 'string' ? c.input : JSON.stringify(c.input);
}
// Model type special handling
if (type === 'Model') {
    content.provider = c.provider ?? 'openai';
    content.model = c.model ?? '';
    if (c.variables) {
        content.variables = c.variables;
    }
    if (s.cost !== undefined) {
        content.cost = s.cost;
    }
    if (c.output !== undefined) {
        const outputObj = typeof c.output === 'object' ? c.output : JSON.parse(JSON.stringify(c.output));
        const parsed = typeof outputObj === 'string' ? JSON.parse(outputObj || '{}') : outputObj;
        if (s.tokens) {
            parsed.tokenUsage = {
                promptTokens: s.tokens.input || 0,
                completionTokens: s.tokens.output || 0,
                totalTokens: s.tokens.total || 0,
            };
        }
        content.output = JSON.stringify(parsed);
    }
}
console.log('\n=== Span Content Structure ===\n');
console.log(JSON.stringify(content, null, 2));
console.log('\n=== Input (parsed) ===\n');
console.log(JSON.stringify(JSON.parse(content.input), null, 2));
console.log('\n=== Output (parsed) ===\n');
console.log(JSON.stringify(JSON.parse(content.output), null, 2));
