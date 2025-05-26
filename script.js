import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0';

// Load LLM and embedder
const generator = await pipeline('text-generation', 'Xenova/distilGPT2', {
  quantized: true,
  progress_callback: x => console.log(`LLM: ${x.loaded}/${x.total}`)
});
const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
  quantized: true,
  progress_callback: x => console.log(`Embedder: ${x.loaded}/${x.total}`)
});
console.log("✅ Models loaded");

// Load and chunk resume
const response = await fetch('resume.txt');
const resumeText = await response.text();
const resumeChunks = resumeText.match(/[^\.?!]+[\.?!]+/g) || [resumeText];

// Helper function to extract embeddings properly
function extractEmbedding(embeddingOutput) {
  // Handle different output formats from the embedder
  if (embeddingOutput && embeddingOutput.data) {
    return Array.from(embeddingOutput.data);
  } else if (Array.isArray(embeddingOutput)) {
    return embeddingOutput.flat(Infinity);
  } else if (embeddingOutput && typeof embeddingOutput.tolist === 'function') {
    return embeddingOutput.tolist().flat(Infinity);
  } else {
    console.error('Unexpected embedding format:', embeddingOutput);
    return [];
  }
}

// Embed chunks
const chunkEmbeddings = [];
for (let chunk of resumeChunks) {
  const embeddingOutput = await embedder(chunk, {
    pooling: 'mean',
    normalize: true
  });
  const embedding = extractEmbedding(embeddingOutput);
  chunkEmbeddings.push(embedding);
  console.log(`Chunk embedded with ${embedding.length} dimensions`);
}
console.log("✅ Chunks embedded:", chunkEmbeddings.length);

// Cosine similarity with better error handling
function cosineSimilarity(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b)) {
    console.error('Non-array vectors passed to cosineSimilarity:', typeof a, typeof b);
    return 0;
  }
  
  if (a.length !== b.length) {
    console.error(`Vector dimension mismatch: ${a.length} vs ${b.length}`);
    return 0;
  }
  
  if (a.length === 0 || b.length === 0) {
    console.error('Empty vectors passed to cosineSimilarity');
    return 0;
  }
  
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  
  if (normA === 0 || normB === 0) {
    console.error('Zero norm vector encountered');
    return 0;
  }
  
  return dot / (normA * normB);
}

// Retrieve top-K relevant chunks
function findRelevantChunks(queryEmbedding, k = 3) {
  const scoredChunks = resumeChunks.map((chunk, idx) => {
    const score = cosineSimilarity(queryEmbedding, chunkEmbeddings[idx]);
    return { chunk, score };
  });
  return scoredChunks
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map(c => c.chunk);
}

// Short-term memory (last 3 Q&A)
const memory = [];
function updateMemory(userQ, botA) {
  memory.push({ userQ, botA });
  if (memory.length > 3) memory.shift();
}

function buildPrompt(question, contextChunks) {
  const context = contextChunks.join(' ');
  const memoryText = memory.map(m => `Q: ${m.userQ}\nA: ${m.botA}`).join('\n');
  return `${memoryText}\nContext: ${context}\n\nQ: ${question}\nA:`;
}

// Main handler with better error handling
async function handleQuestion() {
  const input = document.getElementById('questionInput');
  const answerBox = document.getElementById('answer');
  const question = input.value.trim();
  
  if (!question) return;
  
  answerBox.textContent = 'Thinking...';
  
  try {
    const qEmbeddingOutput = await embedder(question, { 
      pooling: 'mean', 
      normalize: true 
    });
    const qEmbedding = extractEmbedding(qEmbeddingOutput);
    
    if (qEmbedding.length === 0) {
      throw new Error('Failed to generate question embedding');
    }
    
    console.log(`Question embedded with ${qEmbedding.length} dimensions`);
    
    const relevantChunks = findRelevantChunks(qEmbedding);
    const prompt = buildPrompt(question, relevantChunks);
    
    const result = await generator(prompt, {
      max_new_tokens: 100,
      temperature: 0.7
    });
    
    const output = result[0].generated_text.split('A:').pop().trim();
    updateMemory(question, output);
    answerBox.textContent = output;
    
  } catch (error) {
    console.error('Error handling question:', error);
    answerBox.textContent = 'Sorry, I encountered an error processing your question.';
  }
}

// Voice input
const voiceButton = document.getElementById('voiceButton');
if (voiceButton) {
  voiceButton.addEventListener('click', () => {
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-US';
    recognition.start();
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      document.getElementById('questionInput').value = transcript;
      handleQuestion();
    };
  });
}

// Text input listener
const askButton = document.getElementById('askButton');
if (askButton) {
  askButton.addEventListener('click', handleQuestion);
}
