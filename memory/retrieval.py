"""
Memory Retrieval Interface

The bridge between the memory layer and the coach LLM.

Responsibilities:
- Decide when to retrieve from LTM (threshold-based trigger)
- Format retrieved memories into an LLM-readable context block
- Combine STM context window + LTM retrieval into one prompt payload
- Expose a single `build_context()` call that main.py / coach_llm.py uses
"""

from typing import List, Dict, Optional, Tuple

from memory.models import MemoryEntry
from memory.stm_buffer import STMBuffer
from memory.vector_db import LongTermMemory


class MemoryRetriever:
    """
    Assembles the full memory context for each LLM call.

    Usage
    -----
    retriever = MemoryRetriever(stm, ltm)

    # On each user turn, after fusion:
    context = retriever.build_context(fused_result)

    # Pass context["messages"] to LLM as conversation history
    # Pass context["ltm_context"] as an injected system block
    """

    def __init__(
        self,
        stm:                   STMBuffer,
        ltm:                   LongTermMemory,
        ltm_retrieval_top_k:   int   = 3,
        min_arousal_to_trigger: float = 0.55,   # retrieve from LTM when user is emotionally activated
    ):
        self.stm               = stm
        self.ltm               = ltm
        self.top_k             = ltm_retrieval_top_k
        self.arousal_threshold = min_arousal_to_trigger

    def build_context(self, fused_result) -> Dict:
        """
        Build the full context payload for one LLM call.

        Returns a dict with:
          - "messages":      STM turns formatted for LangChain
          - "ltm_context":   Retrieved LTM memories as a formatted string
                             (inject into system prompt or as a prefix message)
          - "emotional_summary": Current session emotional arc
          - "retrieved_entries": Raw MemoryEntry list (for logging/debugging)
          - "ltm_triggered": bool — whether LTM retrieval fired
        """
        # Always include STM
        messages           = self.stm.to_langchain_messages()
        emotional_summary  = self.stm.get_emotional_summary()

        # Conditionally retrieve from LTM
        ltm_triggered = self._should_retrieve(fused_result)
        retrieved     = []

        if ltm_triggered:
            retrieved = self.ltm.retrieve(
                query_text=fused_result.text,
                valence=fused_result.emotional_valence,
                arousal=fused_result.emotional_arousal,
                top_k=self.top_k,
            )

        ltm_context = self._format_ltm_context(retrieved)

        return {
            "messages":           messages,
            "ltm_context":        ltm_context,
            "emotional_summary":  emotional_summary,
            "retrieved_entries":  [e for e, _ in retrieved],
            "ltm_triggered":      ltm_triggered,
        }

    def build_system_prompt_insert(self, fused_result) -> str:
        """
        Convenience method, returns a ready-to-inject string for the
        system prompt that summarises both STM emotional arc and LTM memories.

        Example output injected before the LLM system prompt:
        ---
        Current session: user has been mostly anxious (avg arousal 0.72, 1 conflict).

        Relevant past memories:
        - [anxious] I went quiet when she asked about my job (relevance: 0.81)
        - [INSIGHT] Predicted: 'user deflects with humour'... (relevance: 0.74)
        ---
        """
        ctx     = self.build_context(fused_result)
        summary = ctx["emotional_summary"]

        lines = [
            f"Current session: user has been mostly {summary['dominant_emotion']} "
            f"(avg arousal {summary['avg_arousal']:.2f}, "
            f"{summary['conflict_count']} conflict moment(s)).",
        ]

        if ctx["ltm_triggered"] and ctx["ltm_context"]:
            lines.append("\nRelevant past memories:")
            lines.append(ctx["ltm_context"])

        return "\n".join(lines)

    def _should_retrieve(self, fused_result) -> bool:
        """
        Trigger LTM retrieval when:
        - Emotional arousal exceeds threshold (user is activated)
        - OR a prosody/semantic conflict is detected (notable moment)
        - AND there is something in LTM to retrieve
        """
        if len(self.ltm) == 0:
            return False

        arousal_triggered  = fused_result.emotional_arousal >= self.arousal_threshold
        conflict_triggered = fused_result.has_semantic_conflict

        return arousal_triggered or conflict_triggered

    def _format_ltm_context(
        self,
        retrieved: List[Tuple[MemoryEntry, float]],
    ) -> str:
        """Format retrieved memories as a concise string for the LLM."""
        if not retrieved:
            return ""

        lines = []
        for entry, score in retrieved:
            tag   = f"[{entry.entry_type}]" if entry.entry_type != "episodic" else f"[{entry.emotional_label}]"
            lines.append(f"- {tag} {entry.text[:120]} (relevance: {score:.2f})")

        return "\n".join(lines)


def end_session(
    stm:            STMBuffer,
    ltm:            LongTermMemory,
    salience_floor: float = 0.6,
    run_prune:      bool  = True,
) -> Dict:
    """
    Consolidate STM → LTM and optionally prune at session end.

    Call this when the user finishes a session (pre-date or post-date).
    Returns a summary of what was stored and pruned.
    """
    stored = ltm.consolidate_from_stm(stm, salience_threshold=salience_floor)
    pruned = ltm.prune() if run_prune else 0
    stm.clear()

    return {
        "memories_stored": len(stored),
        "memories_pruned": pruned,
        "ltm_total":       len(ltm),
    }