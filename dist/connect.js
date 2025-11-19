"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
// Minimal runner: use gpt-4o-mini (or set to "gpt-5-nano") and print only the model text.
require("dotenv/config");
const gateway_1 = require("@adaline/gateway");
const openai_1 = require("@adaline/openai");
const types_1 = require("@adaline/types");
async function main() {
    const apiKey = process.env.OAI_API_KEY;
    if (!apiKey) {
        console.error('Error: OAI_API_KEY environment variable is not set.');
        process.exit(1);
    }
    const gateway = new gateway_1.Gateway();
    const openai = new openai_1.OpenAI();
    const model = openai.chatModel({
        // Pick one of the two models you requested. Change here to 'gpt-5-nano' if preferred.
        modelName: 'gpt-4.1-nano',
        apiKey,
    });
    const config = (0, types_1.Config)().parse({ temperature: 0.4,
        top_p: 1,
        max_output_tokens: 512
    });
    const messages = [
        { role: 'system', content: [{ modality: 'text', value: 'You are a helpful assistant.' }] },
        { role: 'user', content: [{ modality: 'text', value: 'Write a short four-line poem about Christmas in AAAB rhyme.' }] },
    ];
    let resp;
    try {
        resp = await gateway.completeChat({ model, config, messages });
    }
    catch (err) {
        const status = err?.cause?.status || err?.cause?.data?.status;
        const code = err?.cause?.data?.error?.code;
        if (status === 401 || code === 'invalid_api_key') {
            console.error('Error: Invalid or incorrect API key. Set a valid OAI_API_KEY in your environment.');
            process.exit(1);
        }
        // For other errors, print a short message and exit non-zero.
        console.error('Request failed:', err?.message ?? err);
        process.exit(1);
    }
    // Extract assistant text robustly. The content items may have different shapes
    // depending on provider; try common fields (.value, .text) and fall back to ''
    const assistantMsg = resp?.response?.messages?.[0];
    const contents = assistantMsg?.content ?? [];
    let text = '';
    for (const c of contents) {
        if (!c)
            continue;
        if (typeof c.value === 'string' && c.value.trim()) {
            text = c.value;
            break;
        }
        if (typeof c.text === 'string' && c.text.trim()) {
            text = c.text;
            break;
        }
        // some items may be { type: 'text', text: '...' }
        if (c.type === 'text' && typeof c.text === 'string' && c.text.trim()) {
            text = c.text;
            break;
        }
    }
    // Print only the assistant text (or empty string). Keep stdout clean.
    console.log(text ?? '');
}
main().catch((err) => {
    console.error('Unhandled error:', err?.message ?? err);
    process.exit(1);
});
