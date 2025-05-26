import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0';

// Load LLM and embedder
const generator = await pipeline('text-generation', 'Xenova/distilGPT2');
const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

console.log("✅ Models loaded");

// Load and chunk resume
const response = await fetch('resume.txt');
const resumeText = await response.text();
const resumeChunks = resumeText.match(/[^\.!\?]+[\.!\?]+/g) || [resumeText]; // sentence chunks

// Embed chunks
const chunkEmbeddings = [];
for (let chunk of resumeChunks) {
  const embedding = await embedder(chunk, {
    pooling: 'mean',
    normalize: true
  });
  chunkEmbeddings.push(embedding[0]); // 1D array
}
console.log("Embedding vector length:", chunkEmbeddings[0].length);

// Cosine similarity
function cosineSimilarity(a, b) {
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dot / (normA * normB);
}

// Retrieve top-K relevant chunks
function findRelevantChunks(queryEmbedding, k = 3) {
  const scoredChunks = resumeChunks.map((chunk, idx) => ({
    chunk,
    score: cosineSimilarity(queryEmbedding, chunkEmbeddings[idx])
  }));
  return scoredChunks.sort((a, b) => b.score - a.score).slice(0, k).map(c => c.chunk);
}

// Short-term memory
const memory = [];

function updateMemory(userQ, botA) {
  memory.push({
