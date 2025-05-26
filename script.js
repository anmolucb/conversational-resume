// script.js

import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.10.0';

let model, embedder;
const chatLog = document.getElementById('chatLog');
const answerDiv = document.getElementById('answer');
const askBtn = document.getElementById('askBtn');
const questionInput = document.getElementById('question');
const startVoiceBtn = document.getElementById('startVoice');
const stopVoiceBtn = document.getElementById('stopVoice');

let resumeChunks = [];
let chunkEmbeddings = [];
let messageHistory = [];
let recognition;

async function loadModel() {
  model = await pipeline('text-generation', 'Xenova/distilgpt2');
  embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  console.log('Models loaded');
}

function addChatEntry(role, text) {
  const entry = document.createElement('div');
  entry.classList.add('chat-entry');
  entry.innerHTML = `<span class="${role}">${role === 'user' ? '🧑' : '🤖'} ${role}:</span> ${text}`;
  chatLog.appendChild(entry);
  chatLog.scrollTop = chatLog.scrollHeight;
}

async function loadResumeChunks() {
  const response = await fetch('resume.txt');
  const raw = await response.text();
  resumeChunks = raw.split(/\[Chunk\]/).map(c => c.trim()).filter(Boolean);
  for (let chunk of resumeChunks) {
    const embedding = await embedder(chunk);
    chunkEmbeddings.push(embedding.data); // If it returns { data: [vector] }
    console.log("Embedding vector length:", chunkEmbeddings[0].length);


  }
}

function cosineSimilarity(a, b) {
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dot / (normA * normB);
}

async function findRelevantChunks(question, k = 3) {
  const qEmbedding = await embedder(question);
  const scores = chunkEmbeddings.map(vec => cosineSimilarity(vec, qEmbedding.data[0]));
  const topKIndices = scores
    .map((score, i) => ({ score, i }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map(entry => entry.i);
  return topKIndices.map(i => resumeChunks[i]);
}

function buildPrompt(chunks, question) {
  const history = messageHistory.slice(-4).map(m => `${m.role.toUpperCase()}: ${m.content}`).join('\n');
  return `You are a helpful assistant answering questions about Anmol Gupta's resume.\n\nResume Context:\n${chunks.join('\n')}\n\n${history}\nUser: ${question}\nAnswer:`;
}

async function handleQuestion(question) {
  addChatEntry('user', question);
  answerDiv.innerText = 'Thinking...';
  const topChunks = await findRelevantChunks(question);
  const prompt = buildPrompt(topChunks, question);

  const output = await model(prompt, { max_new_tokens: 150, temperature: 0.7 });
  const response = output[0].generated_text.split('Answer:')[1].trim();

  answerDiv.innerText = response;
  addChatEntry('bot', response);
  messageHistory.push({ role: 'user', content: question });
  messageHistory.push({ role: 'bot', content: response });
}

askBtn.onclick = () => {
  const question = questionInput.value.trim();
  if (question) {
    handleQuestion(question);
    questionInput.value = '';
  }
};

questionInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') askBtn.click();
});

// WebSpeech API for Voice Input
function initSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    alert('Web Speech API not supported in this browser.');
    return;
  }
  recognition = new SpeechRecognition();
  recognition.lang = 'en-US';
  recognition.interimResults = false;

  recognition.onresult = e => {
    const transcript = e.results[0][0].transcript;
    questionInput.value = transcript;
    askBtn.click();
  };

  recognition.onerror = e => {
    alert('Speech recognition error: ' + e.error);
  };

  recognition.onend = () => {
    startVoiceBtn.disabled = false;
    stopVoiceBtn.disabled = true;
  };
}

startVoiceBtn.onclick = () => {
  recognition.start();
  startVoiceBtn.disabled = true;
  stopVoiceBtn.disabled = false;
};

stopVoiceBtn.onclick = () => {
  recognition.stop();
};

initSpeechRecognition();
await loadModel();
await loadResumeChunks();
