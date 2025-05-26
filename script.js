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
  memory.push({ userQ, botA });
  if (memory.length > 3) memory.shift(); // keep last 3 turns
}

// Handle user question
async function handleQuestion() {
  const question = document.getElementById('questionInput').value.trim();
  if (!question) return;

  const answerDiv = document.getElementById('answer');
  answerDiv.innerText = "Thinking...";

  const queryEmbedding = await embedder(question, {
    pooling: 'mean',
    normalize: true
  });

  const relevantChunks = findRelevantChunks(queryEmbedding[0]);

  const memoryContext = memory.map(m => `User: ${m.userQ}\nBot: ${m.botA}`).join("\n");
  const context = relevantChunks.join(" ");
  const prompt = `${memoryContext}\nBased on this resume info: ${context}\nUser: ${question}\nBot:`;

  const output = await generator(prompt, {
    max_new_tokens: 100,
    temperature: 0.7
  });

  const answer = output[0].generated_text.split("Bot:").pop().trim();
  answerDiv.innerText = answer;
  updateMemory(question, answer);
}

// Button click
document.getElementById('askButton').addEventListener('click', handleQuestion);

// Voice input (WebSpeech API)
const voiceButton = document.getElementById('voiceButton');
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognition = new SpeechRecognition();
  recognition.lang = 'en-US';
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  voiceButton.onclick = () => {
    recognition.start();
    voiceButton.innerText = "🎙️ Listening...";
  };

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    document.getElementById('questionInput').value = transcript;
    voiceButton.innerText = "🎤 Ask by Voice";
    handleQuestion();
  };

  recognition.onerror = () => {
    voiceButton.innerText = "🎤 Ask by Voice";
    alert("Voice recognition failed. Try again.");
  };
} else {
  voiceButton.style.display = 'none'; // browser doesn't support speech API
}

