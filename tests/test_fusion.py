from agent.fusion_engine import ModalityFusionEngine

engine = ModalityFusionEngine()

# --- Option A: Test with hardcoded text + prosody (no mic needed) ---
test_cases = [
    ("I'm really excited about tonight!", "excited/confident"),
    ("Yeah, I'm fine with whatever",      "hesitant/sad"),
    ("I'm so angry I can't think straight", "assertive"),
    ("I don't know, I'm just really scared", "hesitant/sad"),
]

for transcript, prosody in test_cases:
    result = engine.fuse(transcript, prosody)
    print(f"\nText    : {transcript}")
    print(f"Prosody : {prosody}")
    print(f"Label   : {result.emotional_label}  (V={result.emotional_valence:+.2f}, A={result.emotional_arousal:.2f})")
    print(f"Conflict: {result.has_semantic_conflict}")
    if result.conflict_note:
        print(f"Note    : {result.conflict_note}")