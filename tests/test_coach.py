"""
Coach LLM test — no microphone needed.
Tests all three modes with mock fused results.
Requires Ollama running locally: `ollama serve`
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.coach_llm import DatingCoach, CoachMode
from memory.stm_buffer import STMBuffer
from memory.vector_db import LongTermMemory
from memory.retrieval import MemoryRetriever


class MockFused:
    def __init__(self, text, label, valence, arousal, confidence=0.9, conflict=False, note=""):
        self.text                  = text
        self.emotional_label       = label
        self.emotional_valence     = valence
        self.emotional_arousal     = arousal
        self.fusion_confidence     = confidence
        self.has_semantic_conflict = conflict
        self.conflict_note         = note


def run():
    print("=" * 60)
    print("Coach LLM — mode tests")
    print("=" * 60)

    stm = STMBuffer(session_id="test_session")
    ltm = LongTermMemory(storage_path="/tmp/test_coach_ltm")
    retriever = MemoryRetriever(stm, ltm)

    # Seed LTM with a past memory so retrieval has something to surface
    ltm.store_episodic(
        "I froze when she asked about my last relationship",
        emotional_label="anxious", valence=-0.7, arousal=0.85, salience=0.9
    )

    # ── General mode ──────────────────────────────────────────────
    print("\n[1] GENERAL MODE")
    print("─" * 60)
    coach = DatingCoach(mode=CoachMode.GENERAL)

    fused = MockFused(
        "Yeah I'm fine with whatever, I don't really care",
        label="neutral", valence=-0.4, arousal=0.3,
        conflict=True,
        note="Valence gap 0.96 — trusting prosody"
    )
    stm.add_user_turn(fused)
    context = retriever.build_context(fused)
    response = coach.respond(fused, context)
    print(f"User    : \"{fused.text}\"")
    print(f"Emotion : {fused.emotional_label} (conflict: {fused.has_semantic_conflict})")
    print(f"Coach   : {response}\n")

    # ── Pre-date roleplay mode ─────────────────────────────────────
    print("\n[2] PRE-DATE ROLEPLAY MODE")
    print("─" * 60)
    coach.set_mode(CoachMode.PREDATE)

    fused2 = MockFused(
        "I work in software, it's pretty boring honestly",
        label="neutral", valence=-0.1, arousal=0.35
    )
    stm.add_user_turn(fused2)
    context2 = retriever.build_context(fused2)
    response2 = coach.respond(fused2, context2)
    print(f"User    : \"{fused2.text}\"")
    print(f"Emotion : {fused2.emotional_label}")
    print(f"Coach   : {response2}\n")

    # ── Post-date reflection mode ──────────────────────────────────
    print("\n[3] POST-DATE REFLECTION MODE")
    print("─" * 60)

    # Add a couple more turns to give the reflection something to work with
    stm.add_user_turn(MockFused(
        "I got really nervous when she asked about my ex",
        label="anxious", valence=-0.7, arousal=0.85
    ))
    stm.add_user_turn(MockFused(
        "She laughed at my joke and I felt so much better",
        label="excited", valence=0.8, arousal=0.75
    ))

    final_context = retriever.build_context(MockFused(
        "Overall I think it went okay but I froze a few times",
        label="neutral", valence=0.0, arousal=0.5
    ))
    reflection = coach.reflect(final_context)
    print(f"Reflection:\n{reflection}\n")

    print("=" * 60)
    print("✅ Coach tests complete")


if __name__ == "__main__":
    run()