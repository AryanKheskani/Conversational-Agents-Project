# Dating Coach — Conversational Agent

A multimodal, privacy-first conversational agent that acts as a dating coach. It listens to what you say *and* how you say it, fuses those signals together, remembers past sessions, and responds with short spoken coaching advice.

All heavy processing runs locally on your machine — no cloud APIs, no data leaves your device.

---

## Table of Contents

- [What it does](#what-it-does)
- [Key concepts explained](#key-concepts-explained)
- [Architecture overview](#architecture-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the app](#running-the-app)
- [Coaching modes](#coaching-modes)
- [Project structure](#project-structure)
- [Session transcripts](#session-transcripts)
- [Running tests](#running-tests)

---

## What it does

You hold a button to speak. The agent:

1. Transcribes what you said (Whisper, runs locally)
2. Analyses the *tone* of your voice — whether you sound hesitant, excited, assertive, etc.
3. Fuses the meaning of your words with your vocal tone to build a richer picture of how you're feeling
4. Checks its memory for anything relevant from past sessions
5. Generates a short coaching response (1–2 sentences) via a local LLM
6. Speaks the response aloud and displays it word by word in sync with the voice

---

## Key concepts explained

### Semantic analysis

Semantic analysis looks at the **meaning of your words**. A pre-trained emotion classifier (DistilRoBERTa) reads your transcript and determines the emotional content — whether your words express joy, fear, sadness, excitement, and so on.

This produces two scores based purely on *what* you said, ignoring *how* you said it:

- **Valence** — how positive or negative the content is (−1 to +1)
- **Arousal** — how calm or activated / energised (0 to 1)

### Prosodic analysis

Prosody refers to the **non-verbal qualities of speech** — pitch, loudness, rhythm, and pace. Even without understanding words, you can often tell if someone is nervous, confident, or excited just from how they sound.

The agent uses Librosa to extract two signals from your audio:

- **Intensity (RMS energy)** — how loud you are; a proxy for confidence or agitation
- **Pitch variation (F0 standard deviation)** — how much your voice rises and falls; a proxy for emotional expressiveness

These combine into one of four tone labels:

| Tone | Volume | Pitch variation |
|---|---|---|
| excited / confident | high | high |
| assertive | high | low |
| thoughtful | low | high |
| hesitant / sad | low | low |

### Modality fusion

Fusion combines the semantic and prosodic signals into a single unified representation. When both signals agree — your words sound happy *and* your voice sounds excited — the result is high-confidence.

When they conflict — for example, you say "I'm fine" in a flat, hesitant voice — the agent flags a **semantic-prosodic conflict**. In conflict cases, prosody is weighted more heavily (70% prosody, 30% semantic) on the assumption that tone is harder to fake than words.

### Short-term memory (STM)

A sliding window of the last 10 turns within the current session. Each turn is scored by **salience** — a weighted combination of arousal, absolute valence, whether a conflict was detected, and fusion confidence. High-salience turns are passed to the LLM as priority context.

### Long-term memory (LTM)

At the end of each session, high-salience turns are consolidated into a persistent FAISS vector database stored on disk. Each memory is embedded using a 384-dimensional sentence model with an affective suffix (e.g. `[feeling: negative, energy: high]`) so that retrieval is sensitive to both meaning and emotional state.

Memories decay over time using an Ebbinghaus-inspired forgetting curve modulated by salience — something emotionally significant decays much more slowly than a passing remark. During a session, LTM retrieval is triggered when **arousal ≥ 0.55** or a **semantic-prosodic conflict** is detected.

---

## Architecture overview

```
Microphone input
      │
      ├── Whisper ASR ──────────────────► transcript (text)
      │
      └── Librosa prosody ──────────────► tone label
                │
                ▼
        Modality Fusion Engine
        semantic + prosodic → FusedRepresentation
        (valence, arousal, conflict flag, confidence)
                │
                ▼
        Memory Layer
        ├── STM: add turn, compute salience
        └── LTM: retrieve if triggered (FAISS cosine × retention curve)
                │
                ▼
        Coach LLM  (Llama 3.2 via Ollama)
        mode-specific system prompt + STM context + retrieved LTM memories
                │
                ▼
        Response (1–2 sentences)
        ├── Displayed word-by-word in browser (in sync with voice)
        └── Spoken via Web Speech API (browser built-in TTS)
```

---

## Requirements

### System dependencies

- **Python 3.10+**
- **Ollama** — local LLM runtime: [https://ollama.com](https://ollama.com)
- A working **microphone**
- **ffmpeg** — required by Whisper for audio decoding

  ```bash
  # macOS
  brew install ffmpeg

  # Ubuntu / Debian
  sudo apt install ffmpeg
  ```

### Python packages

All listed in `requirements.txt`. Key ones:

| Package | Purpose |
|---|---|
| openai-whisper | Speech-to-text (runs locally) |
| librosa | Pitch and intensity extraction |
| transformers | HuggingFace emotion classifier |
| sentence-transformers | Text embeddings for long-term memory |
| faiss-cpu | Vector similarity search |
| pyaudio | Microphone recording |
| soundfile | Audio file I/O |
| requests | Ollama HTTP client |
| flask-socketio | WebSocket server for the browser UI |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AryanKheskani/Conversational-Agents-Project.git

cd Conversational-Agents-Project
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> The first run will automatically download several models:
> - Whisper `
> - DistilRoBERTa emotion classifier
> - all-MiniLM-L6-v2 sentence embeddings (~90 MB)

### 4. Install and set up Ollama

```bash
# macOS — install via Homebrew
brew install ollama

# Pull the Llama 3.2 model used by the coach (~2 GB)
ollama pull llama3.2
```

For other platforms see [https://ollama.com/download](https://ollama.com/download).

### 5. PyAudio troubleshooting

If `pip install pyaudio` fails, install PortAudio first:

```bash
# macOS
brew install portaudio && pip install pyaudio

# Ubuntu / Debian
sudo apt install portaudio19-dev && pip install pyaudio
```

---

## Running the app

### Step 1 — Start Ollama (required every time)

```bash
ollama serve
```

Leave this running in a terminal in the background.

### Step 2 — Start the web UI

```bash
python3 -m ui.server
```

Then open [http://localhost:5050](http://localhost:5050) in your browser.

**Setup screen options:**

| Option | What it does |
|---|---|
| Mode | Select General, Pre-date, or Post-date coaching |
| Tone analysis | On = prosody + semantic fusion. Off = semantic only |
| Voice | Select the accent/voice for the agent's spoken replies |

Click **Begin session**, then hold **Space** (or the microphone button) to speak. Release to send.

Click **End session** when done — a reflection is generated and your transcript is saved to the `transcripts/` folder.

### Alternative — command-line mode

```bash
python3 main.py
```

Prompts you for mode and prosody preference in the terminal. Push-to-talk via **Space**.

---

## Coaching modes

| Mode | When to use |
|---|---|
| **General** | Ongoing support — building confidence, processing feelings, talking through situations |
| **Pre-date** | Before a date — preparation, managing nerves, conversation ideas |
| **Post-date** | After a date — reflection on what went well, what felt off, patterns over time |

---

## Project structure

```
Conversational-Agents-Project/
├── agent/
│   ├── coach_llm.py          # Ollama client + DatingCoach + mode-specific prompts
│   └── fusion_engine.py      # Semantic + prosodic fusion, conflict detection
├── memory/
│   ├── models.py             # MemoryTurn, MemoryEntry, salience scoring
│   ├── stm_buffer.py         # Short-term memory (10-turn sliding window)
│   ├── embedder.py           # Affect-aware embeddings + forgetting curve
│   ├── vector_db.py          # Long-term memory (FAISS + metadata JSON)
│   ├── retrieval.py          # Context builder, end-of-session consolidation
│   └── ltm_store/            # Persisted FAISS index + metadata (auto-created)
├── perception/
│   ├── audio_asr.py          # Whisper speech-to-text + push-to-talk recording
│   └── audio_prosody.py      # Librosa pitch + intensity → tone label
├── ui/
│   ├── server.py             # Flask + SocketIO WebSocket bridge
│   └── templates/
│       └── index.html        # Single-page UI (emotion display, memory sidebar, PTT)
├── transcripts/              # Per-session JSON transcripts (auto-created on session end)
├── tests/
│   ├── test_fusion.py
│   ├── test_memory.py
│   └── test_coach.py         # Requires Ollama running
├── main.py                   # CLI entry point
├── requirements.txt
└── dating-coach-readme.md
```

---

## Session transcripts

Every session is saved as a JSON file in `transcripts/` when you end the session. The filename is the session ID (a Unix timestamp).

Example:

```json
{
  "session_id": "session_1711234567",
  "mode": "general",
  "use_prosody": true,
  "turns": [
    {
      "role": "user",
      "text": "I've got a date this Friday and I'm really nervous",
      "mode": "prosody+semantic",
      "emotion": "Anxious",
      "valence": -0.42,
      "arousal": 0.71
    },
    {
      "role": "agent",
      "text": "That nervousness is actually a good sign — it means you care. What's the one thing worrying you most?"
    }
  ]
}
```

When **Tone analysis is off**, user turns use `"mode": "semantic"` and omit `valence` and `arousal`.

---

## Running tests

Tests do not require a microphone:

```bash
python3 -m tests.test_fusion    # fusion engine
python3 -m tests.test_memory    # STM + LTM layer
python3 -m tests.test_coach     # coach LLM — requires ollama serve
```
