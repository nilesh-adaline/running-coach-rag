"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
// Quick test to verify span output structure
require("dotenv/config");
const observability_1 = require("./observability");
const fetchDeployedPrompt_1 = require("./fetchDeployedPrompt");
async function testSpanOutput() {
    const trace = (0, observability_1.createTrace)('Test Span Output');
    // Test embedding span with cost
    (0, observability_1.addSpan)(trace, {
        name: 'test_embedding',
        status: 'success',
        startedAt: Date.now(),
        endedAt: Date.now() + 1000,
        content: {
            type: 'Embeddings',
            input: { model: 'text-embedding-3-small', texts: ['test'] },
            output: { embeddings: [[0.1, 0.2]], dimensions: 2 },
        },
        promptId: fetchDeployedPrompt_1.PROMPT_ID,
        cost: 0.00001,
        latency: 1000,
    });
    // Test LLM span with tokens and cost
    (0, observability_1.addSpan)(trace, {
        name: 'test_llm',
        status: 'success',
        startedAt: Date.now(),
        endedAt: Date.now() + 2000,
        content: {
            type: 'Model',
            provider: 'openai',
            model: 'gpt-4.1-mini',
            input: { model: 'gpt-4.1-mini', messages: [{ role: 'user', content: 'test' }] },
            output: { message: { role: 'assistant', content: 'response' } },
        },
        promptId: fetchDeployedPrompt_1.PROMPT_ID,
        cost: 0.0005,
        latency: 2000,
        tokens: {
            input: 10,
            output: 20,
            total: 30,
        },
    });
    console.log('\n=== Trace Structure ===');
    console.log(JSON.stringify(trace, null, 2));
    console.log('\n=== Submitting to API ===');
    await (0, observability_1.submitTrace)(trace);
}
testSpanOutput().catch(console.error);
