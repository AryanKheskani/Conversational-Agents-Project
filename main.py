import os
import time
from perception.audio_asr import record_audio, transcribe_audio
from perception.audio_prosody import analyze_prosody
from agent.fusion_engine import ModalityFusionEngine

def main():
    print("Welcome to the Multimodal Dating Coach Agent!")
    print("Press Ctrl+C at any time to stop.\n")

    engine = ModalityFusionEngine()
    
    try:
        while True:
            # 1. Record Audio (using the record_audio function from audio_asr.py)
            audio_file = record_audio(record_seconds=5)
            
            # 2. Extract Transcription (Semantics)
            print("⏳ Transcribing...")
            text_transcript = transcribe_audio(audio_file)
            
            # 3. Extract Prosody (Affect/Tone)
            print("⏳ Analyzing tone...")
            tone = analyze_prosody(audio_file)
            
            # Fuse transcript + tone into unified emotional representation
            result = engine.fuse(text_transcript, tone)
            print("\n" + "="*50)
            print(f"Transcript : \"{text_transcript}\"")
            print(f"Tone       : [{tone}]")
            print(f"Emotion    : {result.emotional_label} "
                  f"(V={result.emotional_valence:+.2f}, A={result.emotional_arousal:.2f})")
            if result.has_semantic_conflict:
                print(f"⚠️  Conflict  : {result.conflict_note}")
            print(f"Confidence : {result.fusion_confidence:.2f}")
            print("="*50 + "\n")
            
            # Clean up the temporary recording file
            if os.path.exists(audio_file):
                os.remove(audio_file)
                
            # Brief pause before the next iteration
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nStopping Multimodal Dating Coach Agent... Goodbye!")

if __name__ == "__main__":
    main()
