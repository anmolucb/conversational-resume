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
  console.log('Raw embedding output:', embeddingOutput);
  console.log('Type:', typeof embeddingOutput);
  console.log('Is array:', Array.isArray(embeddingOutput));
  
  // Handle Xenova transformer tensor output
  if (embeddingOutput && typeof embeddingOutput === 'object') {
    // Check if it's a tensor with data property
    if (embeddingOutput.data) {
      console.log('Found .data property, length:', embeddingOutput.data.length);
      return Array.from(embeddingOutput.data);
    }
    // Check if it has tolist method
    if (typeof embeddingOutput.tolist === 'function') {
      const result = embeddingOutput.tolist();
      console.log('tolist() result:', result);
      return Array.isArray(result) ? result.flat(Infinity) : [result];
    }
    // Check if it's already an array-like object
    if (embeddingOutput.length !== undefined) {
      console.log('Array-like object, length:', embeddingOutput.length);
      return Array.from(embeddingOutput);
    }
    // Check for nested structure
    if (embeddingOutput[0] && Array.isArray(embeddingOutput[0])) {
      console.log('Nested array structure detected');
      return embeddingOutput[0];
    }
  }
  
  // Handle direct array
  if (Array.isArray(embeddingOutput)) {
    console.log('Direct array, flattening');
    return embeddingOutput.flat(Infinity);
  }
  
  console.error('Could not extract embedding from:', embeddingOutput);
  return [];
}

// Embed chunks
const chunkEmbeddings = [];
console.log('Starting to embed chunks...');
for (let i = 0; i < resumeChunks.length; i++) {
  const chunk = resumeChunks[i];
  console.log(`\nEmbedding chunk ${i + 1}/${resumeChunks.length}`);
  console.log('Chunk text:', chunk.substring(0, 100) + '...');
  
  const embeddingOutput = await embedder(chunk, {
    pooling: 'mean',
    normalize: true
  });
  
  const embedding = extractEmbedding(embeddingOutput);
  console.log('Extracted embedding length:', embedding.length);
  console.log('First few values:', embedding.slice(0, 5));
  
  if (embedding.length === 0) {
    console.error(`Failed to extract embedding for chunk ${i + 1}`);
    continue;
  }
  
  chunkEmbeddings.push(embedding);
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

// Fallback function to extract answers directly from text when LLM fails
function extractDirectAnswer(question, chunks) {
  const lowerQuestion = question.toLowerCase();
  const combinedText = chunks.join(' ').toLowerCase();
  
  // Simple pattern matching for common questions
  if (lowerQuestion.includes('where') && (lowerQuestion.includes('work') || lowerQuestion.includes('job'))) {
    // Look for company names or work locations
    const workPatterns = [
      /works? at ([^.!?]+)/i,
      /employed by ([^.!?]+)/i,
      /company[:\s]+([^.!?]+)/i,
      /organization[:\s]+([^.!?]+)/i
    ];
    
    for (const pattern of workPatterns) {
      const match = chunks.join(' ').match(pattern);
      if (match && match[1]) {
        return `Anmol works at ${match[1].trim()}.`;
      }
    }
  }
  
  if (lowerQuestion.includes('what') && lowerQuestion.includes('do')) {
    // Look for job titles or roles
    const rolePatterns = [
      /position[:\s]+([^.!?]+)/i,
      /role[:\s]+([^.!?]+)/i,
      /title[:\s]+([^.!?]+)/i,
      /job[:\s]+([^.!?]+)/i
    ];
    
    for (const pattern of rolePatterns) {
      const match = chunks.join(' ').match(pattern);
      if (match && match[1]) {
        return `Anmol's role is ${match[1].trim()}.`;
      }
    }
  }
  
  // Return the most relevant chunk if no specific pattern matches
  return chunks[0] || "I couldn't find specific information about that in the resume.";
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
  
  // Simple, direct prompt that works better with small models
  return `Context: ${context}\n\nQuestion: ${question}\nComplete answer:`;
}

// Main handler with better error handling
async function handleQuestion() {
  const input = document.getElementById('questionInput');
  const answerBox = document.getElementById('answer');
  const question = input.value.trim();
  
  if (!question) return;
  
  answerBox.textContent = 'Thinking...';
  
  try {
    console.log('\nProcessing question:', question);
    const qEmbeddingOutput = await embedder(question, { 
      pooling: 'mean', 
      normalize: true 
    });
    console.log('Question embedding output received');
    
    const qEmbedding = extractEmbedding(qEmbeddingOutput);
    console.log('Question embedding extracted, length:', qEmbedding.length);
    console.log('Question embedding first few values:', qEmbedding.slice(0, 5));
    
    if (qEmbedding.length === 0) {
      throw new Error('Failed to generate question embedding');
    }
    
    // Verify chunk embeddings are valid before similarity calculation
    console.log('\nChecking chunk embeddings:');
    for (let i = 0; i < chunkEmbeddings.length; i++) {
      const chunkEmb = chunkEmbeddings[i];
      console.log(`Chunk ${i}: length=${chunkEmb?.length}, isArray=${Array.isArray(chunkEmb)}`);
      if (!Array.isArray(chunkEmb) || chunkEmb.length === 0) {
        console.error(`Invalid chunk embedding at index ${i}:`, chunkEmb);
      }
    }
    
    const relevantChunks = findRelevantChunks(qEmbedding);
    console.log('Relevant chunks found:', relevantChunks.length);
    console.log('Relevant chunks:', relevantChunks);
    
    const prompt = buildPrompt(question, relevantChunks);
    console.log('Generated prompt:', prompt);
    
    const result = await generator(prompt, {
      max_new_tokens: 50,
      temperature: 0.1,
      do_sample: false,
      pad_token_id: 50256,
      eos_token_id: 50256
    });
    
    console.log('Raw generation result:', result);
    
    // Extract answer more carefully
    let output = result[0].generated_text;
    
    // Remove the original prompt to get just the generated part
    const promptEnd = output.indexOf('Complete answer:');
    if (promptEnd !== -1) {
      output = output.substring(promptEnd + 'Complete answer:'.length).trim();
    }
    
    // If output is still incomplete or just repeating, try template-based approach
    if (output.length < 10 || output.toLowerCase().includes('the company is a')) {
      // Extract relevant information directly from chunks
      output = extractDirectAnswer(question, relevantChunks);
    }
    
    // Clean up
    output = output.split('\n')[0].trim();
    if (output.length === 0) {
      output = "I couldn't find specific information about that in the resume.";
    }
    
    console.log('Final output:', output);
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
