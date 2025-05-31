// app.js

// --- Configuration ---
const RESUME_PATH = './resume.txt'; // Path to your resume text file
const EMBEDDING_MODEL_NAME = 'Xenova/all-MiniLM-L6-v2';
const SLM_MODEL_NAME = 'Phi-3-mini-4k-instruct-q4f16_1-MLC'; // Fast loading and capable

// --- Global Variables ---
let resumeTextChunks = [];
let resumeEmbeddings = [];
let embeddingPipeline = null;
let llmEngine = null;
let chatInitialized = false;
let recognition = null; // For SpeechRecognition (voice input)
let isListening = false; // To track microphone state
let speechSynthesis = null; // For SpeechSynthesis API (voice output)
let isAudioOutputEnabled = false; // Default TTS to off
let availableVoices = []; // To store available TTS voices

// --- UI Elements ---
const statusDiv = document.getElementById('status');
const chatbox = document.getElementById('chatbox');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const micButton = document.getElementById('micButton');
const audioOutputToggle = document.getElementById('audioOutputToggle'); // Added for TTS

// --- Core Functions ---

/**
 * 0. Load Resume, Chunk, Embed, and Initialize Models on Page Load
 */
async function initializeApp() {
    console.log("App: Initializing...");
    try {
        updateStatus('Loading embedding model...');
        console.log("Embedding model: Loading...");
        embeddingPipeline = await window.pipeline('feature-extraction', EMBEDDING_MODEL_NAME, {
            progress_callback: (progress) => {
                console.log("Embedding Model Progress:", progress);
                updateStatus(`Embedding Model: ${progress.status} (${Math.round(progress.progress || 0)}%)`);
            }
        });
        updateStatus('Embedding model loaded.');
        console.log("Embedding model: Loaded successfully.");

        updateStatus('Fetching resume...');
        console.log("Resume: Fetching from", RESUME_PATH);
        const response = await fetch(RESUME_PATH);
        if (!response.ok) {
            console.error("Resume: Fetch failed with status", response.statusText);
            throw new Error(`Failed to fetch resume: ${response.statusText}`);
        }
        const resumeFullText = await response.text();
        updateStatus('Resume fetched.');
        console.log("Resume: Fetched successfully. Length:", resumeFullText.length);

        console.log("Resume: Chunking text...");
        resumeTextChunks = chunkText(resumeFullText, 400, 50);
        updateStatus(`Resume split into ${resumeTextChunks.length} chunks.`);
        console.log(`Resume: Split into ${resumeTextChunks.length} chunks. First chunk sample:`, resumeTextChunks[0]?.substring(0, 100) + "...");

        updateStatus('Generating embeddings for resume chunks...');
        console.log("Embeddings: Generating for", resumeTextChunks.length, "chunks...");
        const embeddingsTensor = await embeddingPipeline(resumeTextChunks, {
            pooling: 'mean',
            normalize: true
        });
        resumeEmbeddings = embeddingsTensor.tolist();
        updateStatus('Embeddings generated and stored in memory.');
        console.log("Embeddings: Generated successfully. Stored in memory. Example embedding dimension:", resumeEmbeddings[0]?.length);

        updateStatus('Initializing Language Model (this may take a moment)...');
        console.log("LLM: Initializing. Model selected:", SLM_MODEL_NAME);
        const worker = new Worker(new URL('./web-llm-worker.js', import.meta.url), { type: 'module' });
        console.log("LLM: Web worker created.");

        llmEngine = await window.CreateWebWorkerMLCEngine(
            worker,
            SLM_MODEL_NAME,
            {
                initProgressCallback: (progress) => {
                    console.log("LLM Progress:", progress.text, `(${Math.round(progress.progress * 100)}%)`);
                    updateStatus(`LLM: ${progress.text.replace("[System]", "").trim()}`);
                }
            }
        );
        console.log("LLM: Engine created.");

        updateStatus('LLM: Finalizing model setup...');
        console.log("LLM: Reloading model to ensure readiness...");
        await llmEngine.reload(SLM_MODEL_NAME);
        updateStatus('LLM: Model ready.');
        console.log("LLM: Model reloaded and ready.");

        chatInitialized = true;
        console.log("Chat: Initialized.");

        initializeVoiceInput(); // Initialize Speech-to-Text
        initializeAudioOutput(); // Initialize Text-to-Speech

        // Final UI enabling after all initializations
        userInput.disabled = false;
        sendButton.disabled = false;
        if (recognition) {
            micButton.disabled = false;
            console.log("Voice Input: Mic button enabled.");
        } else {
            micButton.disabled = true;
            console.log("Voice Input: Mic button disabled (not supported or initialized).");
        }
        if (speechSynthesis) { // Enable audio toggle if SpeechSynthesis is supported
            audioOutputToggle.disabled = false;
            console.log("Audio Output: Toggle button enabled.");
        } else {
            audioOutputToggle.disabled = true;
            console.log("Audio Output: Toggle button disabled (not supported).");
        }
        updateStatus('I am ready. Ask me anything about my professional experience and I will do my best to answer your questions!');
        addMessageToChatbox('Assistant', 'Hi there! I\'ve read the resume. How can I help you?');
        console.log("App: Fully initialized.");

    } catch (error) {
        console.error('Initialization Error:', error.message, error.stack);
        updateStatus(`Error initializing: ${error.message}`);
        addMessageToChatbox('Assistant', `I couldn't start properly: ${error.message}`);
        userInput.disabled = true;
        sendButton.disabled = true;
        micButton.disabled = true;
        if (audioOutputToggle) audioOutputToggle.disabled = true; // Also disable audio toggle on critical error
    }
}

/**
 * Text Chunking Implementation
 */
function chunkText(text, chunkSize = 500, overlap = 50) {
    const chunks = [];
    let i = 0;
    while (i < text.length) {
        const end = Math.min(i + chunkSize, text.length);
        chunks.push(text.slice(i, end));
        if (end === text.length) break;
        i += chunkSize - overlap;
    }
    return chunks.filter(chunk => chunk.trim().length > 10);
}

/**
 * Single Embedding Generation (for query)
 */
async function generateSingleEmbedding(text) {
    if (!embeddingPipeline) {
        console.error("generateSingleEmbedding: Embedding pipeline not initialized.");
        throw new Error("Embedding pipeline not initialized.");
    }
    console.log("Embeddings: Generating for query:", text.substring(0, 50) + "...");
    const result = await embeddingPipeline(text, { pooling: 'mean', normalize: true });
    console.log("Embeddings: Query embedding generated.");
    return result.tolist()[0];
}

/**
 * Initialize Voice Input (Speech-to-Text)
 */
function initializeVoiceInput() {
    console.log("Voice Input: Initializing...");
    const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognitionAPI) {
        updateStatus('Voice recognition not supported by this browser.');
        if(micButton) micButton.disabled = true;
        console.warn('Voice Input: Speech Recognition API not found.');
        return;
    }

    recognition = new SpeechRecognitionAPI();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    console.log("Voice Input: SpeechRecognition object created with lang", recognition.lang);

    recognition.onstart = () => {
        isListening = true;
        micButton.classList.add('listening');
        micButton.textContent = '👂';
        updateStatus('Listening...');
        sendButton.disabled = true;
        console.log("Voice Input: Recognition started. Listening...");
    };

    recognition.onresult = (event) => {
        const last = event.results.length - 1;
        const transcript = event.results[last][0].transcript.trim();
        userInput.value = transcript;
        updateStatus(`Recognized: "${transcript}"`);
        console.log("Voice Input: Result received. Transcript:", transcript);
        if (transcript) {
            console.log("Voice Input: Transcript valid, calling handleUserQuery.");
            handleUserQuery();
        }
    };

    recognition.onerror = (event) => {
        console.error('Voice Input: Speech recognition error -', event.error, event.message);
        let errorMessage = 'Voice input error';
        if (event.error === 'no-speech') {
            errorMessage = 'No speech detected. Please try again.';
        } else if (event.error === 'audio-capture') {
            errorMessage = 'Microphone problem. Ensure it is enabled and working.';
        } else if (event.error === 'not-allowed') {
            errorMessage = 'Microphone access denied. Allow access in browser settings.';
        } else if (event.error === 'network') {
            errorMessage = 'Network error during voice recognition.';
        }
        updateStatus(errorMessage);
    };

    recognition.onend = () => {
        isListening = false;
        micButton.classList.remove('listening');
        micButton.textContent = '🎤';
        if (chatInitialized && !userInput.disabled) {
             sendButton.disabled = false;
        }
        console.log("Voice Input: Recognition ended.");
    };

    micButton.addEventListener('click', () => {
        if (!recognition) {
            console.warn("Voice Input: Mic button clicked but recognition not initialized.");
            return;
        }

        if (isListening) {
            console.log("Voice Input: Mic button clicked while listening. Stopping recognition.");
            recognition.stop();
        } else {
            console.log("Voice Input: Mic button clicked. Starting recognition.");
            try {
                if (chatInitialized) userInput.disabled = false;
                recognition.start();
            } catch (e) {
                console.error("Voice Input: Error starting recognition -", e.name, e.message);
                if (e.name === 'InvalidStateError') {
                    updateStatus('Please wait before trying voice input again.');
                } else {
                    updateStatus('Could not start voice input.');
                }
                isListening = false;
                micButton.classList.remove('listening');
                micButton.textContent = '🎤';
            }
        }
    });
    console.log("Voice Input: Event listeners set up.");
}

/**
 * Initialize Audio Output (Text-to-Speech)
 */
function initializeAudioOutput() {
    console.log("Audio Output: Initializing...");
    if ('speechSynthesis' in window) {
        speechSynthesis = window.speechSynthesis;
        console.log("Audio Output: SpeechSynthesis API found.");

        const populateVoices = () => {
            availableVoices = speechSynthesis.getVoices();
            console.log("Audio Output: Available voices:", availableVoices.length > 0 ? availableVoices.length + " voices found." : "None initially, may load async.");
            // Log details of a few voices if available
            if (availableVoices.length > 0) {
                console.log("Audio Output: Example voices:", availableVoices.slice(0, 5).map(v => ({name: v.name, lang: v.lang})));
            }
        };
        populateVoices(); // Initial call
        if (speechSynthesis.onvoiceschanged !== undefined) { // Check if event is supported
            speechSynthesis.onvoiceschanged = populateVoices; // Update when voice list changes
        }

        if (audioOutputToggle) {
            audioOutputToggle.disabled = false; // Enable the toggle button
            audioOutputToggle.addEventListener('click', () => {
                isAudioOutputEnabled = !isAudioOutputEnabled;
                if (isAudioOutputEnabled) {
                    audioOutputToggle.textContent = '🔊 Audio On';
                    audioOutputToggle.style.backgroundColor = '#28a745'; // Green
                    console.log("Audio Output: Enabled by user.");
                    speakText("Audio output enabled.");
                } else {
                    audioOutputToggle.textContent = '🔇 Audio Off';
                    audioOutputToggle.style.backgroundColor = '#6c757d'; // Grey
                    console.log("Audio Output: Disabled by user.");
                    if (speechSynthesis.speaking) {
                        speechSynthesis.cancel();
                        console.log("Audio Output: Cancelled ongoing speech.");
                    }
                }
            });
        } else {
            console.warn("Audio Output: Toggle button not found in DOM.");
        }
    } else {
        updateStatus('Text-to-speech not supported by this browser.');
        if (audioOutputToggle) audioOutputToggle.disabled = true;
        console.warn('Audio Output: SpeechSynthesis API not found.');
    }
}

/**
 * Speak the provided text using SpeechSynthesis
 * @param {string} text - The text to speak
 */
function speakText(text) {
    if (!isAudioOutputEnabled || !speechSynthesis || !text || text.trim() === "") {
        // console.log("Audio Output: Skipped speaking (disabled, not supported, or empty text).");
        return;
    }
    console.log("Audio Output: Attempting to speak -", text.substring(0, 50) + "...");

    if (speechSynthesis.speaking || speechSynthesis.pending) {
        speechSynthesis.cancel(); 
        console.log("Audio Output: Cancelled previous/pending speech for new utterance.");
        // A brief pause after cancel can sometimes help ensure the next utterance starts cleanly
        // This is especially true on some browsers/platforms.
        setTimeout(() => proceedWithSpeech(text), 50); 
    } else {
        proceedWithSpeech(text);
    }
}

function proceedWithSpeech(text) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US'; // Default language

    // Attempt to select a suitable voice
    const preferredVoice = availableVoices.find(voice => voice.lang === 'en-US' && voice.default) || 
                           availableVoices.find(voice => voice.lang === 'en-US');
    if (preferredVoice) {
        utterance.voice = preferredVoice;
        console.log("Audio Output: Using voice -", preferredVoice.name, `(${preferredVoice.lang})`);
    } else if (availableVoices.length > 0) {
        utterance.voice = availableVoices[0]; // Fallback to the first available voice
         console.log("Audio Output: Preferred 'en-US' voice not found, using first available voice -", availableVoices[0].name, `(${availableVoices[0].lang})`);
    } else {
        console.log("Audio Output: No voices available, using browser default for lang", utterance.lang);
    }
    
    utterance.rate = 1.0; 
    utterance.pitch = 1.0;

    utterance.onstart = () => {
        console.log("Audio Output: Speech started for utterance.");
    };
    utterance.onend = () => {
        console.log("Audio Output: Speech ended for utterance.");
    };
    utterance.onerror = (event) => {
        console.error("Audio Output: Speech synthesis error -", event.error, "for text:", event.utterance.text.substring(0, 50) + "...");
    };
    
    speechSynthesis.speak(utterance);
}


/**
 * Cosine Similarity Function
 */
function cosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * User Query Handling
 */
async function handleUserQuery() {
    const query = userInput.value.trim();
    if (!query || !chatInitialized) {
        console.warn("handleUserQuery: Query is empty or chat not initialized. Aborting.");
        return;
    }
    console.log("Query Handling: Started for query -", query);

    addMessageToChatbox('User', query);
    userInput.value = '';
    userInput.disabled = true;
    sendButton.disabled = true;
    if (recognition && !isListening && micButton) micButton.disabled = true;
    updateStatus('Thinking...');

    try {
        const queryEmbedding = await generateSingleEmbedding(query);
        console.log("Query Handling: Query embedding generated.");

        const N = 3;
        console.log("Query Handling: Retrieving top", N, "relevant chunks.");
        const similarities = resumeEmbeddings.map((chunkEmb, index) => ({
            index: index,
            text: resumeTextChunks[index],
            similarity: cosineSimilarity(queryEmbedding, chunkEmb)
        }));

        similarities.sort((a, b) => b.similarity - a.similarity);
        const relevantChunks = similarities.slice(0, N).map(sim => resumeTextChunks[sim.index]);

        console.log("Query Handling: Relevant chunks found:", relevantChunks.map(chunk => ({
            text: chunk.substring(0, 100) + "...",
        })));

        const context = relevantChunks.join('\n---\n');
        const prompt = `You are a helpful assistant representing Anmol Gupta. Please do not deviate from the the resume and stick to whatever information you are able able to find. Please talk as a 1st person, that is refer yourself as I. Be respectful and professional. Do not use any bad words or give opinion on political or religious matters or anything else that could cause problem. Answer question Based ONLY on the resume information. Do not mix information or context while answering the question the use asks. If the information is not in the context, say "I will not be able to answer that question at the moment."\n\nCONTEXT:\n${context}\n\nQUESTION: ${query}\n\nANSWER:`;
        console.log("Query Handling: Constructed prompt for LLM. Length:", prompt.length);

        let fullResponse = "";
        console.log("Query Handling: Sending prompt to LLM for completion...");
        const stream = await llmEngine.chat.completions.create({
            messages: [{ role: "user", content: prompt }],
            stream: true,
        });
        console.log("Query Handling: LLM stream initiated.");

        let firstChunk = true;
        let assistantMessageDivSpan = null;

        for await (const chunk of stream) {
            const deltaContent = chunk.choices[0]?.delta?.content;
            if (deltaContent) {
                fullResponse += deltaContent;
                if (firstChunk) {
                    assistantMessageDivSpan = addMessageToChatbox('Assistant', deltaContent, true);
                    firstChunk = false;
                } else if (assistantMessageDivSpan) {
                    assistantMessageDivSpan.innerHTML += deltaContent.replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\n/g, '<br>');
                    chatbox.scrollTop = chatbox.scrollHeight;
                }
            }
        }
        console.log("Query Handling: LLM stream finished. Full response length:", fullResponse.length);

        let responseToSpeak = "";
        if (firstChunk && fullResponse === "") {
             responseToSpeak = "I have no relevant information to share.";
             addMessageToChatbox('Assistant', responseToSpeak);
             console.log("Query Handling: LLM provided no relevant information or an empty response.");
        } else if (firstChunk && fullResponse !== "") { 
            responseToSpeak = fullResponse;
            addMessageToChatbox('Assistant', responseToSpeak);
            console.log("Query Handling: LLM response added (non-streamed fallback).");
        } else if (!firstChunk && fullResponse !== "") {
            responseToSpeak = fullResponse;
            // Message already added via streaming
            console.log("Query Handling: Streamed response complete.");
        }
        speakText(responseToSpeak); // Speak the final determined response

    } catch (error) {
        console.error('Query Handling: Error -', error.message, error.stack);
        const errorReply = `Sorry, I am facing a problem. Please try again later.`;
        addMessageToChatbox('Assistant', errorReply);
        speakText(errorReply); // Speak the generic error reply
    } finally {
        userInput.disabled = false;
        sendButton.disabled = false;
         if (recognition && micButton) micButton.disabled = false;
        updateStatus('I am ready. Ask me anything about my professional experience and I will do my best to answer your questions!');
        userInput.focus();
        console.log("Query Handling: Finished. UI re-enabled.");
    }
}

// --- Helper UI Functions ---
function updateStatus(message) {
    if(statusDiv) statusDiv.textContent = message;
}

/**
 * Adds a message to the chatbox with appropriate styling and wrapper for alignment.
 * @param {string} sender - 'User' or 'Assistant'
 * @param {string} message - The message content
 * @param {boolean} returnSpanForStreaming - If true, returns the inner span for live updates
 * @returns {HTMLSpanElement | null} - The span element if returnSpanForStreaming is true, otherwise null
 */
function addMessageToChatbox(sender, message, returnSpanForStreaming = false) {
    if (!chatbox) {
        console.error("addMessageToChatbox: chatbox element not found!");
        return null;
    }

    const messageWrapper = document.createElement('div');
    messageWrapper.classList.add('message-wrapper');

    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender.toLowerCase() + '-message');

    if (sender.toLowerCase() === 'user') {
        messageWrapper.classList.add('user-message-wrapper');
    } else {
        messageWrapper.classList.add('assistant-message-wrapper');
    }

    const cleanMessage = message.replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const messageSpan = document.createElement('span');
    messageSpan.innerHTML = cleanMessage.replace(/\n/g, '<br>');

    messageDiv.appendChild(messageSpan);
    messageWrapper.appendChild(messageDiv);
    chatbox.appendChild(messageWrapper);
    chatbox.scrollTop = chatbox.scrollHeight;

    if (returnSpanForStreaming) {
        return messageSpan;
    }
    return null;
}

// --- Event Listeners ---
if (sendButton) {
    sendButton.addEventListener('click', handleUserQuery);
} else {
    console.error("Send button not found during event listener setup.");
}
if (userInput) {
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter' && !sendButton.disabled) {
            handleUserQuery();
        }
    });
} else {
    console.error("User input not found during event listener setup.");
}
console.log("App: Event listeners for send button and user input potentially attached.");

// --- Initialize ---
initializeApp();
