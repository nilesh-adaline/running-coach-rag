// basics/createDB.tsx
// Simple PDF -> Pinecone uploader using Adaline Gateway + OpenAI adapter
// - Keeps the Adaline Gateway setup used elsewhere in this project
// - Exports two functions for reuse in `response.tsx`:
//     export async function queryVectorDB(question: string, topK?: number)
//     export async function addDocumentToDB(pdfPath: string, metadata?: any)

import 'dotenv/config';
import { readdir, readFile } from 'fs/promises';
import path from 'path';
// pdf-parse has no bundled types; require and treat as any for simplicity
const pdfParse: any = require('pdf-parse');
import { Pinecone } from '@pinecone-database/pinecone';
import { OpenAI } from '@adaline/openai';
import { Gateway } from '@adaline/gateway';
import { Config } from '@adaline/types';

// -------------------------
// Config (from environment)
// -------------------------
const PINECONE_INDEX = process.env.PINECONE_INDEX_NAME || 'adx-test-app';
const PINECONE_ENDPOINT = process.env.PINECONE_ENDPOINT || 'https://adx-test-app-1-b5c087b.svc.aped-4627-b74a.pinecone.io';
const PINECONE_DIMENSION = Number(process.env.PINECONE_DIMENSION || '1024');

const OPENAI_API_KEY = process.env.OPENAI_API_KEY || process.env.OAI_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY || '';

if (!OPENAI_API_KEY) throw new Error('OPENAI_API_KEY (or OAI_API_KEY) must be set in .env');
if (!PINECONE_API_KEY) throw new Error('PINECONE_API_KEY must be set in .env');

// -------------------------
// Clients
// -------------------------
const pinecone = new Pinecone({ apiKey: PINECONE_API_KEY });
const openai = new OpenAI();
const gateway = new Gateway();

// -------------------------
// Utilities
// -------------------------
// Deterministic block-averaging projection to match index dimension
function projectToDim(src: number[], targetDim: number): number[] {
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

function getIndex() {
  return pinecone.Index(PINECONE_INDEX, PINECONE_ENDPOINT);
}

// -------------------------
// 1) Read PDF and extract text
// -------------------------
export async function extractTextFromPDF(filePath: string): Promise<string> {
  // Read PDF bytes and extract text using pdf-parse
  const data = await readFile(filePath);
  const res = await pdfParse(data);
  const text = (res.text || '').replace(/\r/g, '');
  return text.trim();
}

// -------------------------
// 2) Split text into chunks (sentence-aware, overlap)
// -------------------------
export function splitIntoChunks(text: string, chunkSize = 900, overlap = 150): string[] {
  // Split into sentence-like pieces first to avoid chopping sentences
  const sentences = text.split(/(?<=[.?!])\s+/g);
  const chunks: string[] = [];
  let current = '';

  for (const s of sentences) {
    if ((current + ' ' + s).trim().length <= chunkSize) {
      current = (current + ' ' + s).trim();
    } else {
      if (current.length > 0) {
        chunks.push(current);
        // carry overlap: keep last `overlap` characters as the start of the next chunk
        const startOverlap = Math.max(0, current.length - overlap);
        current = current.slice(startOverlap).trim();
      }
      // if a single sentence is longer than chunkSize, hard-split it
      if (s.length > chunkSize) {
        for (let i = 0; i < s.length; i += chunkSize - overlap) {
          chunks.push(s.slice(i, i + chunkSize));
        }
        current = '';
      } else {
        current = s.trim();
      }
    }
  }
  if (current.length > 0) chunks.push(current);
  return chunks;
}

// -------------------------
// 3) Create embeddings via Adaline Gateway -> OpenAI adapter
// -------------------------
export async function createEmbeddings(texts: string[], batchSize = 100): Promise<number[][]> {
  if (texts.length === 0) return [];
  const out: number[][] = [];
  // OPENAI_API_KEY validated earlier; assert here
  const model = openai.embeddingModel({ modelName: 'text-embedding-3-small', apiKey: OPENAI_API_KEY as string });

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    const config = Config().parse({});

    const resp: any = await gateway.getEmbeddings({
      model,
      config,
      embeddingRequests: { modality: 'text', requests: batch },
    });

    const items = resp?.response?.embeddings ?? [];
    for (let j = 0; j < batch.length; j++) {
      const item = items[j];
      const emb = item?.embedding as number[] | undefined;
      if (!Array.isArray(emb)) {
        // if something went wrong, push a zero-vector placeholder
        out.push(new Array(PINECONE_DIMENSION).fill(0));
      } else if (emb.length !== PINECONE_DIMENSION) {
        // project deterministically to match index dimension
        out.push(projectToDim(emb, PINECONE_DIMENSION));
      } else {
        out.push(emb);
      }
    }
  }
  return out;
}

// -------------------------
// 4) Save to Pinecone
// -------------------------
export async function saveToPinecone(chunks: string[], embeddings: number[][], fileName: string): Promise<void> {
  const index = getIndex();
  const vectors: any[] = [];

  for (let i = 0; i < chunks.length; i++) {
    const id = `${path.basename(fileName, path.extname(fileName))}-chunk-${i}`;
    vectors.push({ id, values: embeddings[i], metadata: { text: chunks[i], source: path.basename(fileName), chunkIndex: i } });
  }

  // Upsert in batches to avoid large single requests
  const batchSize = 50;
  for (let i = 0; i < vectors.length; i += batchSize) {
    const b = vectors.slice(i, i + batchSize);
    await index.upsert(b);
  }
}

// -------------------------
// Exported functions required by your project
// -------------------------
// Query the vector DB for a question (exports matches with metadata)
export async function queryVectorDB(question: string, topK = 5): Promise<any> {
  try {
    const embeddings = await createEmbeddings([question]);
    const q = embeddings[0];
    const index = getIndex();
    const res: any = await index.query({ vector: q, topK, includeMetadata: true });
    return res.matches ?? [];
  } catch (err: any) {
    console.error('queryVectorDB error:', err?.message ?? err);
    return [];
  }
}

// Add a single PDF document to the DB (extract, chunk, embed, save)
export async function addDocumentToDB(pdfPath: string, metadata: any = {}): Promise<void> {
  try {
    console.log(`Processing: ${path.basename(pdfPath)}`);
    const text = await extractTextFromPDF(pdfPath);
    console.log(`  → Extracted ${text.length.toLocaleString()} characters`);

    const chunks = splitIntoChunks(text);
    console.log(`  → Split into ${chunks.length} chunk(s)`);

    const embeddings = await createEmbeddings(chunks);
    console.log(`  → Generated embeddings`);

    // Merge provided metadata into per-vector metadata
    const index = getIndex();
    const vectors: any[] = [];
    for (let i = 0; i < chunks.length; i++) {
      const id = `${path.basename(pdfPath, path.extname(pdfPath))}-chunk-${i}`;
      const md = { text: chunks[i], source: path.basename(pdfPath), chunkIndex: i, ...metadata };
      vectors.push({ id, values: embeddings[i], metadata: md });
    }

    // Upsert
    const batchSize = 50;
    for (let i = 0; i < vectors.length; i += batchSize) {
      const b = vectors.slice(i, i + batchSize);
      await index.upsert(b);
    }

    console.log(`  → Saved to Pinecone`);
    console.log(`✓ Successfully processed: ${path.basename(pdfPath)} (${chunks.length} chunks)`);
  } catch (err: any) {
    console.error(`✗ Failed: ${path.basename(pdfPath)} - ${(err && err.message) || err}`);
  }
}

// -------------------------
// 5) Convenience: process all PDFs in a folder
// -------------------------
export async function processAllPDFs(folderPath: string): Promise<void> {
  const files = (await readdir(folderPath)).filter((f) => f.toLowerCase().endsWith('.pdf'));
  console.log(`Starting PDF processing... Found ${files.length} PDF(s)`);

  let totalChunks = 0;
  for (let i = 0; i < files.length; i++) {
    const fname = files[i];
    console.log(`\nProcessing file ${i + 1} of ${files.length}: ${fname}`);
    const full = path.join(folderPath, fname);
    await addDocumentToDB(full);
    // best-effort: count chunks by reading metadata or re-splitting
    try {
      const text = await extractTextFromPDF(full);
      totalChunks += splitIntoChunks(text).length;
    } catch {
      // ignore
    }
  }

  console.log(`\nDone! Processed ${files.length} PDF(s), ${totalChunks} total chunks`);
}

// If run directly, process `data-pdfs/` folder by default
if (require.main === module) {
  const folder = path.join(process.cwd(), 'data');
  processAllPDFs(folder)
    .then(() => console.log('All PDFs processed'))
    .catch((err) => { console.error('Fatal error:', err); process.exit(1); });
}
