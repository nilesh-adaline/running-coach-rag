import 'dotenv/config';
import { Pinecone } from '@pinecone-database/pinecone';
import { OpenAI } from '@adaline/openai';
import { Gateway } from '@adaline/gateway';
import { Config } from '@adaline/types';
import { readFile, readdir } from 'fs/promises';
const pdfParse: any = require('pdf-parse');
import path from 'path';
import { getUserQuery, getCoachTemplate } from './prompt';
import { PROMPT_ID } from './fetchDeployedPrompt';
import { Trace, addSpan } from './observability';

const PINECONE_INDEX = 'adx-test-app';
const PINECONE_ENDPOINT = 'https://adx-test-app-1-b5c087b.svc.aped-4627-b74a.pinecone.io';
const PINECONE_DIMENSION = 1024;

const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY || '' });
const openai = new OpenAI();
const gateway = new Gateway();

// Deterministic block-averaging projection (matches createDB.tsx logic)
export function projectToDim(src: number[], targetDim: number): number[] {
  const m = src.length;
  if (m === targetDim) return src.slice();
  const out = new Array<number>(targetDim);
  if (m > targetDim) {
    for (let j = 0; j < targetDim; j++) {
      const start = Math.floor((j * m) / targetDim);
      const end = Math.floor(((j + 1) * m) / targetDim) || (start + 1);
      let s = 0;
      for (let i = start; i < end && i < m; i++) s += src[i];
      out[j] = s / Math.max(1, end - start);
    }
  } else {
    for (let j = 0; j < targetDim; j++) {
      const idx = Math.floor((j * m) / targetDim);
      out[j] = src[Math.min(m - 1, idx)];
    }
  }
  return out;
}

export async function getIndex() {
  return pinecone.Index(PINECONE_INDEX, PINECONE_ENDPOINT);
}

// Create embedding for a query via Adaline Gateway
export async function createQueryEmbedding(text: string, trace?: Trace, parentReferenceId?: string): Promise<number[]> {
  const startTime = Date.now();
  const apiKey = process.env.OAI_API_KEY;
  if (!apiKey) throw new Error('OAI_API_KEY is required in the environment');

  const modelName = 'text-embedding-3-small';
  const model = openai.embeddingModel({ modelName, apiKey });
  const config = Config().parse({});
  let finalEmb: number[] = [];
  let status: 'success' | 'error' = 'success';
  let errorMessage = '';
  let originalDimension = 0;

  try {
    const resp: any = await gateway.getEmbeddings({
      model,
      config,
      embeddingRequests: { modality: 'text', requests: [text] },
    });

    const items = resp?.response?.embeddings;
    if (!items || items.length === 0) throw new Error('No embeddings returned from gateway');
    const emb = items[0].embedding as number[];
    if (!Array.isArray(emb)) throw new Error('Unexpected embedding format');

    originalDimension = emb.length;
    // Project to match index dimension
    finalEmb = emb.length !== PINECONE_DIMENSION ? projectToDim(emb, PINECONE_DIMENSION) : emb;
  } catch (error) {
    status = 'error';
    errorMessage = error instanceof Error ? error.message : String(error);
    console.error('Error creating query embedding:', error);
    throw error;
  } finally {
    if (trace) {
        const endTime = Date.now();
        // text-embedding-3-small: $0.00002 per 1K tokens (rough estimate: 1 char ≈ 0.25 tokens)
        const estimatedTokens = Math.ceil(text.length * 0.25);
        const embeddingCost = (estimatedTokens / 1000) * 0.00002;
        
        console.log(`   💰 Embedding cost: $${embeddingCost.toFixed(6)} (${estimatedTokens} tokens)`);
        
        addSpan(trace, {
          name: 'embedding_create',
        status,
        startedAt: startTime,
        endedAt: endTime,
        parentReferenceId,
        content: {
          type: 'Embeddings',
            input: { 
              operation: 'create_query_embedding',
              model: modelName, 
              texts: [text],
              textLength: text.length,
              textWordCount: text.split(/\s+/).length,
              estimatedTokens,
            },
            output: { 
              embeddings: [finalEmb], 
              dimensions: finalEmb.length,
              originalDimension,
              projectedDimension: PINECONE_DIMENSION,
              projectionApplied: originalDimension !== PINECONE_DIMENSION,
              embeddingModel: modelName,
              vectorNorm: Math.sqrt(finalEmb.reduce((sum, val) => sum + val * val, 0)),
              error: errorMessage || undefined,
            },
        },
        promptId: PROMPT_ID,
        latency: endTime - startTime,
        cost: embeddingCost,
      });
    }
  }

  return finalEmb;
}

// Query Pinecone with top-k matches using the full prompt.
export async function retrieveTopK(k = 5, trace?: Trace, fullPromptOverride?: string, parentReferenceId?: string) {
  let fullPrompt = fullPromptOverride;
  if (!fullPrompt) {
    const coach = await getCoachTemplate();
    const userQuery = await getUserQuery();
    fullPrompt = `${coach}\n\nUser request:\n${userQuery}`;
  }

//   console.log(fullPrompt);

  const index = await getIndex();
  const qemb = await createQueryEmbedding(fullPrompt, trace, parentReferenceId);

  const startTime = Date.now();
  let matches: any[] = [];
  let status: 'success' | 'error' = 'success';
  let errorMessage = '';
  let queryExecuted = false;

  try {
    const results: any = await index.query({ vector: qemb, topK: k, includeMetadata: true });
    matches = results.matches ?? [];
    queryExecuted = true;
  } catch (error) {
    status = 'error';
    errorMessage = error instanceof Error ? error.message : String(error);
    console.error('Error retrieving from Pinecone:', error);
    throw error;
  } finally {
    if (trace) {
        const endTime = Date.now();
        
        // Calculate average score and extract metadata info
        const avgScore = matches.length > 0 
          ? matches.reduce((sum, m) => sum + (m.score || 0), 0) / matches.length 
          : 0;
        const scores = matches.map(m => m.score || 0);
        const maxScore = scores.length > 0 ? Math.max(...scores) : 0;
        const minScore = scores.length > 0 ? Math.min(...scores) : 0;
        
        addSpan(trace, {
          name: 'pinecone_query',
        status,
        startedAt: startTime,
        endedAt: endTime,
        parentReferenceId,
        content: {
          type: 'Retrieval',
            input: { 
              operation: 'vector_similarity_search',
              query: fullPrompt as string,
              queryLength: (fullPrompt as string).length,
              queryWordCount: (fullPrompt as string).split(/\s+/).length,
              topK: k,
              indexName: PINECONE_INDEX,
              vectorDimension: qemb.length,
              includeMetadata: true,
            },
          output: { 
              documents: matches,
              retrievedCount: matches.length,
              requestedTopK: k,
              avgSimilarityScore: avgScore,
              maxSimilarityScore: maxScore,
              minSimilarityScore: minScore,
              documentIds: matches.map(m => m.id),
              hasMetadata: matches.every(m => m.metadata),
              queryExecuted,
              indexEndpoint: PINECONE_ENDPOINT,
              error: errorMessage || undefined,
            },
        },
        promptId: PROMPT_ID,
        latency: endTime - startTime,
        cost: 0, // Pinecone cost varies by plan; set to 0 for now
      });
    }
  }

  return matches;
}

// Read a chunk's content from a source file in ./data (supports .mdx and .pdf)
export async function readChunkContent(fileName: string, chunkNum: number, chunkSize = 800) {
  try {
    const filePath = path.join(process.cwd(), 'data', fileName);
    // If PDF, extract text via pdf-parse
    let doc = '';
    if (fileName.toLowerCase().endsWith('.pdf')) {
      const buf = await readFile(filePath);
      try {
        const pdfRes: any = await pdfParse(buf);
        doc = (pdfRes.text || '').replace(/\s+/g, ' ').trim();
      } catch {
        doc = (await readFile(filePath, 'utf8')).replace(/\s+/g, ' ').trim();
      }
    } else {
      const raw = await readFile(filePath, 'utf8');
      const body = raw.replace(/^---[\s\S]*?---/, '').replace(/<!--([\s\S]*?)-->/g, '').trim();
      doc = body.length > 0 ? body.replace(/\s+/g, ' ') : fileName.replace('.mdx', '');
    }

    const start = chunkNum * chunkSize;
    return doc.slice(start, start + chunkSize);
  } catch (err) {
    return '';
  }
}

// Parse metadata from a match to get file and chunk info; fallback to ID
export async function parseMatchMetadata(match: any): Promise<{ fileName?: string; chunkNum?: number }> {
  let fileName: string | undefined;
  let chunkNum: number | undefined;

  if (match.metadata) {
    fileName = (match.metadata.file ?? match.metadata.source ?? match.metadata.filename) as string | undefined;
    chunkNum = (match.metadata.chunk ?? match.metadata.chunkIndex ?? match.metadata.chunk_num) as number | undefined;
  }

  if ((!fileName || chunkNum === undefined) && match.id) {
    const idStr = String(match.id);
    const idMatch = idStr.match(/(.+)-chunk-(\d+)$/);
    if (idMatch) {
      const base = idMatch[1];
      const idx = Number(idMatch[2]);

      try {
        const dataDir = path.join(process.cwd(), 'data');
        const files = await readdir(dataDir);
        const found = files.find((f) => path.basename(f, path.extname(f)) === base || f.startsWith(base));
        if (found) fileName = found;
      } catch {
        // ignore
      }

      if (!fileName) fileName = `${base}.mdx`;
      if (chunkNum === undefined) chunkNum = idx;
    }
  }

  return { fileName, chunkNum };
}

async function main() {
  console.log(`\n🔍 Searching with full prompt (coach template + user query)\n`);
  try {
    console.log('Querying Pinecone...');
    const matches = await retrieveTopK(5);
    console.log(`\n📊 Found ${matches.length} similar vector(s):\n`);

    if (matches.length === 0) {
      console.log('No matches found.');
    } else {
      for (let idx = 0; idx < matches.length; idx++) {
        const match = matches[idx];
        console.log(`${idx + 1}. Vector ID: ${match.id}`);
        console.log(`   Score: ${((match.score ?? 0) * 100).toFixed(2)}%`);

        const { fileName, chunkNum } = await parseMatchMetadata(match);
        console.log(`   File: ${fileName ?? 'unknown'}`);
        console.log(`   Chunk: ${chunkNum ?? 'unknown'}`);

        if (fileName && typeof chunkNum === 'number') {
          const content = await readChunkContent(fileName, chunkNum);
          const preview = content.substring(0, 150).trim();
          console.log(`   Content: "${preview}${content.length > 150 ? '...' : ''}"`);
        } else {
          console.log('   Content: (unavailable)');
        }
        console.log('');
      }
    }

    console.log('✓ Retrieval complete');
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    console.error('Error:', errorMsg);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}