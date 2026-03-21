# Multimodal Dating Coach Agent

An empathetic, privacy-first conversational agent designed to help users prepare for and reflect on romantic social interactions. This system uses a local Large Language Model (Llama 3.2) and a hybrid memory architecture to provide personalized coaching. 

Unlike standard text-based bots, this agent employs **Multimodal Acoustic-Semantic Fusion**, analyzing both *what* the user says (Semantic/Text) and *how* they say it (Affective/Prosody) to detect emotional mismatches and provide deeper, context-aware support.


---

## Features
* **100% Local & Privacy-Compliant:** All audio processing, vector storage, and LLM generation happen locally on-device. No sensitive romantic data is sent to the cloud.
* **Dual-Stream Perception:** Uses OpenAI Whisper for speech-to-text and Librosa for prosodic tone extraction (pitch/intensity).
* **Acoustic-Semantic Disambiguation:** Automatically detects when a user's words contradict their tone of voice (e.g., saying "I'm fine" with an anxious tone) and prompts clarifying questions.
* **Hybrid Memory Architecture:**
  * **Short-Term Memory (STM):** LangChain buffer for immediate conversational context.
  * **Long-Term Memory (LTM):** FAISS vector database for persistent, affect-aware episodic recall across sessions.

---

## How It Works

```
Microphone
    │
    ├──► Whisper (audio_asr.py)       → transcript
    └──► Librosa (audio_prosody.py)   → prosody label
                    │
                    ▼
          Fusion Engine (fusion_engine.py)
          ┌─────────────────────────────┐
          │ HuggingFace emotion model   │
          │ + negation detection        │
          │ + conflict detection        │
          └────────────┬────────────────┘
                       │ FusedRepresentation
                       ▼
          Memory Layer (memory/)
          ┌─────────────────────────────┐
          │ STM: sliding turn buffer    │
          │ LTM: FAISS + forgetting     │
          │ Retriever: affect-aware RAG │
          └────────────┬────────────────┘
                       │ context dict
                       ▼
          Coach LLM (coach_llm.py)
          ┌─────────────────────────────┐
          │ Llama 3.2 via Ollama        │
          │ Mode-specific system prompt │
          └─────────────────────────────┘
                       │
                       ▼
                 Coach response
```

---

## Prerequisites
* Python 3.10+
* A working microphone
* [Ollama](https://ollama.com/) installed on your machine

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone 
cd Conversational-Agents-Project
```

### 2. Install system dependencies (macOS)
```bash
brew install portaudio ffmpeg ollama
```

### 3. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
```

### 4. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 5. Pull the local LLM
```bash
ollama pull llama3.2
```

### 6. Run the agent
```bash
# In one terminal, keep running:
ollama serve

# In another terminal:
python3 main.py
```

You will be prompted to select a coaching mode before the session begins.

---

## Running Tests          

Tests do not require a microphone and can be run independently:
```bash
python3 -m tests.test_fusion    # fusion engine
python3 -m tests.test_memory    # memory layer (STM + LTM)
python3 -m tests.test_coach     # coach LLM (requires ollama serve)
```

---

## Setup Instructions

Conversational-Agents-Project/
│
├── perception/           # Modality Extraction
│   ├── audio_asr.py      # Transcribes speech to text using Whisper
│   └── audio_prosody.py  # Extracts tone/emotion using Librosa
│
├── agent/                # Core Agent Logic
│   ├── fusion_engine.py  # Compares Text vs. Tone for conflict detection
│   └── coach_llm.py      # LangChain connection to local Ollama instance
│
├── memory/
│   ├── models.py             # Shared dataclasses (MemoryTurn, MemoryEntry)
│   ├── stm_buffer.py         # Short-term sliding window buffer
│   ├── embedder.py           # Affect-aware sentence embeddings + forgetting curve
│   ├── vector_db.py          # FAISS long-term store with pruning + predict-calibrate
│   └── retrieval.py          # LTM trigger logic + LLM context assembly
│
├── tests/
│   ├── test_fusion.py        # Fusion engine unit tests
│   ├── test_memory.py        # Memory layer integration tests
│   └── test_coach.py         # Coach LLM mode tests (no mic needed)
│
├── data/                 # Local Storage (Excluded from version control)
│   ├── faiss_index/      # Persistent memory embeddings
│   └── temp_audio/       # Temporary .wav files (overwritten each turn)
│
├── requirements.txt      # Python dependencies
└── main.py               # Main execution loop