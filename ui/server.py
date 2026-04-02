# ui/server.py
"""
WebSocket bridge between the dating coach pipeline and the browser UI.
Run this instead of main.py:  python3 -m ui.server

The existing pipeline (fusion, memory, coach) runs in a background thread.
State is pushed to the browser via Flask-SocketIO events.
"""

import os
import sys
import time
import json
import threading
import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit

from perception.audio_asr import record_audio, transcribe_audio
from perception.audio_prosody import analyze_prosody
from agent.fusion_engine import ModalityFusionEngine
from agent.coach_llm import DatingCoach, CoachMode
from memory.stm_buffer import STMBuffer
from memory.vector_db import LongTermMemory
from memory.retrieval import MemoryRetriever, end_session

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = "dating-coach-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Global session state
_session = {
    "engine": None,
    "stm": None,
    "ltm": None,
    "retriever": None,
    "coach": None,
    "mode": "general",
    "use_prosody": True,
    "running": False,
    "recording": False,
    "session_id": None,
    "transcript": [],
}
_pipeline_thread = None
_shutdown = threading.Event()


# Routes

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


# Socket events (browser → server)

@socketio.on("connect")
def on_connect():
    emit("status", {"msg": "Connected to Dating Coach server"})
    # Send current LTM memory count if session exists
    if _session["ltm"]:
        emit("ltm_count", {"count": len(_session["ltm"])})


@socketio.on("init_session")
def on_init_session(data):
    """Browser sends mode + prosody preference, server initialises pipeline."""
    global _session

    mode_str = data.get("mode", "general")
    use_prosody = data.get("use_prosody", True)

    mode_map = {
        "general":  CoachMode.GENERAL,
        "predate":  CoachMode.PREDATE,
        "postdate": CoachMode.POSTDATE,
    }
    mode = mode_map.get(mode_str, CoachMode.GENERAL)

    session_id = f"session_{int(time.time())}"

    _session["engine"] = ModalityFusionEngine(use_prosody=use_prosody)
    _session["stm"] = STMBuffer(max_turns=10, session_id=session_id)
    _session["ltm"] = LongTermMemory(storage_path="memory/ltm_store")
    _session["retriever"] = MemoryRetriever(_session["stm"], _session["ltm"])
    _session["coach"] = DatingCoach(mode=mode)
    _session["mode"] = mode_str
    _session["use_prosody"] = use_prosody
    _session["running"] = True
    _session["session_id"] = session_id
    _session["transcript"] = []

    _session["ltm"].prune()

    # Send past memories to browser
    past = _get_past_memories()

    emit("session_ready", {
        "session_id": session_id,
        "mode": mode_str,
        "use_prosody": use_prosody,
        "ltm_count": len(_session["ltm"]),
        "past_memories": past,
    })


@socketio.on("start_recording")
def on_start_recording():
    """Browser push-to-talk pressed — start recording in background thread."""
    if not _session["running"]:
        emit("error", {"msg": "Session not initialised. Please configure and start first."})
        return
    if _session["recording"]:
        return

    _session["recording"] = True
    _shutdown.clear()
    emit("recording_started", {})

    t = threading.Thread(target=_run_turn, daemon=True)
    t.start()


@socketio.on("stop_recording")
def on_stop_recording():
    """Browser push-to-talk released — signal recording to stop."""
    # The audio_asr record_audio uses release_event; we signal via file flag
    # For simplicity, the turn thread reads a shared event
    _shutdown.set()


@socketio.on("end_session")
def on_end_session():
    """Browser requests session wrap-up."""
    if not _session["running"]:
        return

    stm = _session["stm"]
    ltm = _session["ltm"]

    # Generate reflection if not in postdate mode
    if _session["mode"] != "postdate" and stm and len(stm.get_user_turns()) > 0:
        emit("status", {"msg": "Generating session reflection..."})
        last_turn = stm.get_user_turns()[-1]

        class _Proxy:
            def __init__(self, t):
                self.text                  = t.text
                self.emotional_label       = t.emotional_label
                self.emotional_valence     = t.emotional_valence
                self.emotional_arousal     = t.emotional_arousal
                self.fusion_confidence     = t.fusion_confidence
                self.has_semantic_conflict = t.has_conflict
                self.conflict_note         = ""

        ctx        = _session["retriever"].build_context(_Proxy(last_turn))
        reflection = _session["coach"].reflect(ctx)
        emit("reflection", {"text": reflection})

    summary = end_session(stm, ltm, salience_floor=0.4, run_prune=True)
    _session["running"] = False

    # Save transcript to JSON
    if _session["transcript"]:
        transcripts_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "transcripts"
        )
        os.makedirs(transcripts_dir, exist_ok=True)
        transcript_data = {
            "session_id":  _session["session_id"],
            "mode":        _session["mode"],
            "use_prosody": _session["use_prosody"],
            "turns":       _session["transcript"],
        }
        path = os.path.join(transcripts_dir, f"{_session['session_id']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)

    emit("session_ended", {
        "memories_stored": summary["memories_stored"],
        "memories_pruned": summary["memories_pruned"],
        "ltm_total":       summary["ltm_total"],
    })


# Pipeline turn

def _run_turn():
    """Run one full perception → fusion → memory → coach turn."""
    audio_file = None
    try:
        # Record
        socketio.emit("status", {"msg": "🔴 Recording... release to stop"})
        audio_file = record_audio(external_stop=_shutdown, auto_start=True)

        if audio_file is None:
            socketio.emit("recording_stopped", {})
            return

        socketio.emit("recording_stopped", {})

        # Transcribe
        socketio.emit("status", {"msg": "Transcribing..."})
        transcript = transcribe_audio(audio_file)

        if not transcript.strip():
            socketio.emit("status", {"msg": "No speech detected."})
            return

        socketio.emit("transcript", {"text": transcript})

        # Prosody
        tone = None
        if _session["use_prosody"]:
            socketio.emit("status", {"msg": "Analysing tone..."})
            tone = analyze_prosody(audio_file)
            socketio.emit("tone", {"label": tone})

        # Fuse
        result = _session["engine"].fuse(transcript, tone)
        _session["stm"].add_user_turn(result)

        socketio.emit("emotion", {
            "label":      result.emotional_label,
            "valence":    result.emotional_valence,
            "arousal":    result.emotional_arousal,
            "confidence": result.fusion_confidence,
            "conflict":   result.has_semantic_conflict,
            "conflict_note": result.conflict_note,
            "salience":   _session["stm"].get_user_turns()[-1].salience,
        })

        # Record user turn in transcript
        user_entry = {
            "role":   "user",
            "text":   transcript,
            "mode":   "prosody+semantic" if _session["use_prosody"] else "semantic",
            "emotion": result.emotional_label,
        }
        if _session["use_prosody"]:
            user_entry["valence"] = round(result.emotional_valence, 3)
            user_entry["arousal"] = round(result.emotional_arousal, 3)
        _session["transcript"].append(user_entry)

        # Memory retrieval
        context = _session["retriever"].build_context(result)
        if context["ltm_triggered"] and context["retrieved_entries"]:
            memories = [
                {
                    "type":  e.entry_type,
                    "label": e.emotional_label,
                    "text":  e.text[:120],
                    "score": round(s, 3),
                }
                for e, s in zip(
                    context["retrieved_entries"],
                    [r[1] for r in _session["ltm"].retrieve(
                        result.text,
                        valence=result.emotional_valence,
                        arousal=result.emotional_arousal,
                        top_k=3,
                    )]
                )
            ]
            socketio.emit("memories_retrieved", {"memories": memories})

        # Coach response
        socketio.emit("coach_thinking", {})
        response = _session["coach"].respond(result, context, auto_speak=False)
        _session["stm"].add_agent_turn(response)

        # Record agent turn in transcript
        _session["transcript"].append({"role": "agent", "text": response})

        # Send full response — TTS and word-reveal are handled client-side in sync
        socketio.emit("coach_response_start", {"text": response})
        socketio.emit("status", {"msg": "Hold SPACE to speak  |  Q to quit"})

    except Exception as e:
        socketio.emit("error", {"msg": str(e)})
    finally:
        _session["recording"] = False
        if audio_file and os.path.exists(audio_file):
            os.remove(audio_file)


def _get_past_memories():
    ltm = _session.get("ltm")
    if not ltm:
        return []
    return [
        {
            "id":    e.id,
            "type":  e.entry_type,
            "label": e.emotional_label,
            "text":  e.text[:140],
            "session_id": e.session_id,
        }
        for e in ltm._entries[-20:]  # last 20
    ]


# Entry point

if __name__ == "__main__":
    print("Dating Coach UI — http://localhost:5050")
    socketio.run(app, host="0.0.0.0", port=5050, debug=False)