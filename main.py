import os
import time
import signal
import warnings
warnings.filterwarnings("ignore")

from perception.audio_asr import record_audio, transcribe_audio
from perception.audio_prosody import analyze_prosody
from agent.fusion_engine import ModalityFusionEngine
from memory.stm_buffer import STMBuffer
from memory.vector_db import LongTermMemory
from memory.retrieval import MemoryRetriever, end_session


# Flag checked between operations
_shutdown = False

def _handle_sigint(sig, frame):
    global _shutdown
    _shutdown = True

signal.signal(signal.SIGINT, _handle_sigint)


def main():
    print("Welcome to the Multimodal Dating Coach Agent!")
    print("Press Ctrl+C at any time to stop.\n")

    engine = ModalityFusionEngine()
    session_id = f"session_{int(time.time())}"
    stm       = STMBuffer(max_turns=10, session_id=session_id)
    ltm       = LongTermMemory(storage_path="memory/ltm_store")
    retriever = MemoryRetriever(stm, ltm)

    # Prune stale memories from previous sessions at startup
    ltm.prune()

    print(f"Session: {session_id}")
    print(f"LTM: {len(ltm)} memories loaded from previous sessions\n")

    while not _shutdown:
        audio_file = None
        try:
            # 1. Record Audio (using the record_audio function from audio_asr.py)
            audio_file = record_audio(record_seconds=5)
            if _shutdown: break
            
            # 2. Extract Transcription (Semantics)
            print("⏳ Transcribing...")
            text_transcript = transcribe_audio(audio_file)
            if _shutdown: break

            # 3. Extract Prosody (Affect/Tone)
            print("⏳ Analyzing tone...")
            tone = analyze_prosody(audio_file)
            if _shutdown: break

            # 4. Fuse transcript + tone
            result = engine.fuse(text_transcript, tone)

            # 5. Update STM
            stm.add_user_turn(result)

            # 6. Check if LTM retrieval is triggered
            context = retriever.build_context(result)

            print("\n" + "="*50)
            print(f"Transcript : \"{text_transcript}\"")
            print(f"Tone       : [{tone}]")
            print(f"Emotion    : {result.emotional_label} "
                  f"(V={result.emotional_valence:+.2f}, A={result.emotional_arousal:.2f})")
            if result.has_semantic_conflict:
                print(f"⚠️  Conflict  : {result.conflict_note}")
            print(f"Confidence : {result.fusion_confidence:.2f}")
            print(f"Salience   : {stm.get_user_turns()[-1].salience:.2f}")

            # 8. Show retrieved memories if LTM fired
            if context["ltm_triggered"] and context["retrieved_entries"]:
                print("\n🧠 Relevant past memories:")
                for entry in context["retrieved_entries"]:
                    tag = f"[{entry.entry_type}]" if entry.entry_type != "episodic" \
                          else f"[{entry.emotional_label}]"
                    print(f"   {tag} {entry.text[:80]}")

            print("="*50 + "\n")
            
        finally:
            # Clean up the temporary recording file
            if audio_file and os.path.exists(audio_file):
                os.remove(audio_file)

        # Brief pause before the next iteration
        time.sleep(1)

    print("\n\nWrapping up session...")
    summary = end_session(stm, ltm, salience_floor=0.6, run_prune=True)
    print(f"💾 Saved {summary['memories_stored']} memories to long-term storage")
    print(f"🗑️  Pruned {summary['memories_pruned']} stale memories")
    print(f"📦 LTM total: {summary['ltm_total']} memories")
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
