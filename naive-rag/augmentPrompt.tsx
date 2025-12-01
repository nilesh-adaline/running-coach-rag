// basics/augmentPrompt.tsx
// Build an augmented prompt by combining the coach template, user query, retrieved context,
// and additional context. This module exports a single function used by `response.tsx`
// to assemble the final prompt sent to the LLM.

import { getCoachTemplate, getUserQuery, getFullPrompt } from './prompt';
import { PROMPT_ID } from './fetchDeployedPrompt';
import { Trace, addSpan } from './observability';

// Create an augmented prompt string. Combines the full prompt (coach template + user query)
// with retrieved document snippets to provide the LLM with both instructions and context.
export async function buildAugmentedPrompt(
  contextSnippets: string[],
  trace?: Trace,
  coachTemplateOverride?: string,
  userQueryOverride?: string,
  parentReferenceId?: string
): Promise<string> {
  const startTime = Date.now();
  let coach: string;
  let userQuery: string;
  if (coachTemplateOverride !== undefined && userQueryOverride !== undefined) {
    coach = coachTemplateOverride;
    userQuery = userQueryOverride;
  } else {
    const both = await getFullPrompt(trace);
    coach = both.coachTemplate;
    userQuery = both.userQuery;
  }

  let out = '';
  out += `${coach}\n\n`;
  out += `User Request:\n${userQuery}\n\n`;
  out += `Context from retrieval (only include when relevant):\n`;
  for (let i = 0; i < contextSnippets.length; i++) {
    out += `--- snippet ${i + 1} ---\n${contextSnippets[i]}\n\n`;
  }

  out += `Instructions for assistant:\nUse ONLY the requested sections from the template. Keep answers tight and metric. If pain or heat is a concern, add a short caution.\n`;

  // Log augmentation span if trace is provided
  if (trace) {
    const endTime = Date.now();
    
    // Calculate statistics about the augmented prompt
    const coachWordCount = coach.split(/\s+/).length;
    const userQueryWordCount = userQuery.split(/\s+/).length;
    const contextBlock = contextSnippets.join('\n\n');
    const contextWordCount = contextBlock.split(/\s+/).length;
    const totalWordCount = out.split(/\s+/).length;
    
    addSpan(trace, {
      name: 'prompt_augmentation',
      status: 'success',
      startedAt: startTime,
      endedAt: endTime,
      parentReferenceId,
      content: {
        type: 'Function',
        input: { 
          operation: 'augment_prompt_with_retrieval_context',
          snippetsIncluded: contextSnippets.length,
          coachTemplateLength: coach.length,
          userQueryLength: userQuery.length,
          contextSnippetsLengths: contextSnippets.map(s => s.length),
          totalContextLength: contextSnippets.reduce((sum, s) => sum + s.length, 0),
        },
        output: {
          augmentedPrompt: out,
          augmentedPromptLength: out.length,
          components: ['coach_template', 'user_query', 'retrieval_context', 'instructions'],
          componentLengths: {
            coachTemplate: coach.length,
            userQuery: userQuery.length,
            retrievalContext: contextBlock.length,
          },
          wordCounts: {
            coachTemplate: coachWordCount,
            userQuery: userQueryWordCount,
            retrievalContext: contextWordCount,
            total: totalWordCount,
          },
          snippetsProcessed: contextSnippets.length,
          estimatedTokens: Math.ceil(out.length / 4),
        },
      },
      promptId: PROMPT_ID,
      latency: endTime - startTime,
      cost: 0, // No cost for string concatenation
    });
  }

  return out;
}

// Lightweight CLI runner to verify augmentation works standalone.
// Usage: npx ts-node basics/augmentPrompt.tsx [k]
// If [k] is provided and retrieval is available, it will attempt to fetch k snippets.
if (require.main === module) {
  (async () => {
    try {
      const k = Number(process.argv[2] ?? '0');
      let snippets: string[] = [];

      if (k > 0) {
        try {
          const { retrieveTopK, readChunkContent, parseMatchMetadata } = await import('./retrieve');
          const matches: any[] = await retrieveTopK(k);

          for (const match of matches) {
            const { fileName, chunkNum } = await parseMatchMetadata(match);
            if (fileName && typeof chunkNum === 'number') {
              const content = await readChunkContent(fileName, chunkNum);
              if (content) snippets.push(content.substring(0, 600));
            }
          }
        } catch (e) {
          // Retrieval not available; fall back to dummy snippet
        }
      }

      if (snippets.length === 0) {
        snippets = [
          'This is a sample context snippet to verify augmentation works without retrieval.',
        ];
      }

  const augmented = await buildAugmentedPrompt(snippets);
  console.log('\n=== Augmented Prompt ===');
  console.log(augmented);
  console.log(`\n\n[Length: ${augmented.length} chars | Snippets: ${snippets.length}]`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error('augmentPrompt error:', msg);
      process.exit(1);
    }
  })();
}
