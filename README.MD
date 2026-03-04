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

## Prerequisites
* Python 3.9+
* A working microphone
* [Ollama](https://ollama.com/) installed on your machine

---

## Setup Instructions

**1. Clone or Initialize the Repository**
```bash
mkdir Conversational-Agents-Project
cd Conversational-Agents-Project

**2. Create a Virtual Environment**
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

**3. Install Dependencies**
pip install -r requirements.txt

**4. Setup the Local LLM (Llama 3.2)**
ollama run llama3.2

**5. Start the Agent**
python main.py

**6. Brew Installations**
brew install portaudio
brew install ollama
brew install ffmpeg


## Setup Instructions

Conversational-Agents-Project/
│
├── perception/           # Modality Extraction
│   ├── audio_asr.py      # Transcribes speech to text using Whisper
│   └── audio_prosody.py  # Extracts tone/emotion using Librosa
│
├── memory/               # Hybrid Memory Architecture
│   ├── vector_db.py      # FAISS setup for Long-Term Memory
│   ├── stm_buffer.py     # LangChain buffer for Short-Term Memory
│   └── retrieval.py      # Affect-aware retrieval logic
│
├── agent/                # Core Agent Logic
│   ├── fusion_engine.py  # Compares Text vs. Tone for conflict detection
│   └── coach_llm.py      # LangChain connection to local Ollama instance
│
├── data/                 # Local Storage (Excluded from version control)
│   ├── faiss_index/      # Persistent memory embeddings
│   └── temp_audio/       # Temporary .wav files (overwritten each turn)
│
├── requirements.txt      # Python dependencies
└── main.py               # Main execution loop