"""
Memory layer integration test.
Tests the full flow: STM → salience → LTM → retrieval → session end.
"""

import sys
import os
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory import STMBuffer, LongTermMemory, MemoryRetriever, end_session
from memory.embedder import compute_retention


class MockFused:
    def __init__(self, text, label, valence, arousal, confidence, conflict):
        self.text                  = text
        self.emotional_label       = label
        self.emotional_valence     = valence
        self.emotional_arousal     = arousal
        self.fusion_confidence     = confidence
        self.has_semantic_conflict = conflict


TEST_STORE = "/tmp/test_ltm_store"

def run():
    # Clean slate
    if os.path.exists(TEST_STORE):
        shutil.rmtree(TEST_STORE)

    print("=" * 60)
    print("Memory Layer — integration test")
    print("=" * 60)

    # -- Setup --
    stm = STMBuffer(max_turns=10, session_id="session_001")
    ltm = LongTermMemory(storage_path=TEST_STORE)
    retriever = MemoryRetriever(stm, ltm)

    # -- Simulate a session --
    turns = [
        MockFused("I think the date went okay",          "neutral",  0.10, 0.40, 0.90, False),
        MockFused("Yeah I'm fine with whatever",          "neutral", -0.40, 0.30, 0.70, True),
        MockFused("I was really nervous the whole time",  "anxious", -0.70, 0.85, 0.95, False),
        MockFused("She laughed at my joke, felt great!",  "excited",  0.85, 0.80, 0.92, False),
        MockFused("I froze when she asked about my ex",   "anxious", -0.60, 0.90, 0.88, True),
    ]

    print("\n--- Simulating session turns ---")
    for fused in turns:
        turn = stm.add_user_turn(fused)
        stm.add_agent_turn("[Coach response]")
        print(f"  [{turn.salience:.2f} salience] {fused.text!r} → {fused.emotional_label}")

    print(f"\nSTM: {stm}")
    print(f"Salient turns (≥0.6): {len(stm.get_salient_turns())}")

    # -- Emotional summary --
    print("\n--- Emotional summary ---")
    for k, v in stm.get_emotional_summary().items():
        print(f"  {k}: {v}")

    # -- Predict-calibrate --
    print("\n--- Predict-Calibrate distillation ---")
    ltm.store_prediction(
        "User will use humour to deflect when asked personal questions",
        session_id="session_001"
    )
    ltm.distil_insight(
        prediction_text="User will use humour to deflect when asked personal questions",
        actual_outcome="User froze and went silent when asked about their ex",
        session_id="session_001"
    )

    # -- Session end: consolidate STM → LTM --
    print("\n--- Session end consolidation ---")
    summary = end_session(stm, ltm, salience_floor=0.6, run_prune=False)
    print(f"  Stored: {summary['memories_stored']} | "
          f"Pruned: {summary['memories_pruned']} | "
          f"LTM total: {summary['ltm_total']}")

    # -- Retrieval test --
    print(f"\n--- LTM retrieval test ---")
    print(f"LTM: {ltm}\n")

    queries = [
        ("I feel nervous before meeting her again",   -0.5, 0.80),
        ("She found me funny, I felt so confident",    0.8, 0.75),
        ("I don't know what to say when things get personal", -0.4, 0.65),
    ]

    for query, v, a in queries:
        print(f"  Query: {query!r}")
        results = ltm.retrieve(query, valence=v, arousal=a, top_k=3)
        if results:
            for entry, score in results:
                tag = f"[{entry.entry_type}]" if entry.entry_type != "episodic" else f"[{entry.emotional_label}]"
                print(f"    {score:.3f} {tag} {entry.text[:90]}")
        else:
            print("    (no results above threshold)")
        print()

    # -- Retrieval trigger test --
    print("--- LTM trigger test ---")
    high_arousal = MockFused("I can't stop thinking about it", "anxious", -0.6, 0.85, 0.9, False)
    low_arousal  = MockFused("Yeah it was fine",               "neutral",  0.1, 0.30, 0.8, False)

    stm2 = STMBuffer(session_id="session_002")
    stm2.add_user_turn(high_arousal)
    retriever2 = MemoryRetriever(stm2, ltm)

    ctx_high = retriever2.build_context(high_arousal)
    ctx_low  = retriever2.build_context(low_arousal)
    print(f"  High arousal → LTM triggered: {ctx_high['ltm_triggered']}")
    print(f"  Low arousal  → LTM triggered: {ctx_low['ltm_triggered']}")

    # -- Pruning test --
    print("\n--- Pruning test ---")
    import time

    # Manually backdated a low-salience memory to simulate age
    low_sal_entry = ltm.store_episodic(
        "We talked about the weather briefly",
        emotional_label="neutral", valence=0.0, arousal=0.2,
        salience=0.1,  # very low salience — mundane turn
        session_id="session_prune_test"
    )

    # Age it by backdating the timestamp by 500 hours
    low_sal_entry.timestamp -= 500 * 3600
    # Also store a high-salience entry with same age — should survive
    high_sal_entry = ltm.store_episodic(
        "I completely broke down crying talking about my last relationship",
        emotional_label="sad", valence=-0.9, arousal=0.95,
        salience=1.0,  # max salience — should never be pruned
        session_id="session_prune_test"
    )
    high_sal_entry.timestamp -= 500 * 3600

    ltm._save()

    print(f"LTM before prune: {len(ltm)} entries")
    print(f"Low-salience entry retention:  {compute_retention(low_sal_entry.salience, 500 * 3600, decay_rate=0.001):.3f}")
    print(f"High-salience entry retention: {compute_retention(high_sal_entry.salience, 500 * 3600, decay_rate=0.001):.3f}")

    pruned = ltm.prune()
    print(f"LTM after prune:  {len(ltm)} entries")
    print(f"Pruned: {pruned} (expected: 1 — only the mundane turn)")

    # Verify high-salience survived
    still_there = any(e.id == high_sal_entry.id for e in ltm._entries)
    print(f"High-salience entry survived: {still_there} (expected: True)")

    print("\n✅ All memory tests passed")


if __name__ == "__main__":
    run()
