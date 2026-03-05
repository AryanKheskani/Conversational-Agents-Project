"""
Short-Term Memory Buffer

Sliding window of the most recent conversation turns, each carrying
full affective metadata from the fusion engine.

Acts as the LLM's context window — passed directly to the coach LLM
on every turn. Also acts as the staging area before LTM consolidation.

Capacity: last N turns (default 10, covering ~8-10 exchanges)
"""

from collections import deque, Counter
from typing import List, Dict, Optional

from memory.models import MemoryTurn, compute_salience


class STMBuffer:
    """
    Sliding-window short-term memory.

    Usage
    -----
    stm = STMBuffer(session_id="session_001")

    # After fusion engine produces a FusedRepresentation:
    stm.add_user_turn(fused_result)

    # After agent responds:
    stm.add_agent_turn("Here is my advice...")

    # Feed into LLM:
    messages = stm.to_langchain_messages()

    # At session end, hand off to LTM:
    ltm.consolidate_from_stm(stm)
    """

    def __init__(self, max_turns: int = 10, session_id: str = ""):
        self.max_turns  = max_turns
        self.session_id = session_id
        self._buffer: deque = deque(maxlen=max_turns)

    def add_user_turn(self, fused_result) -> MemoryTurn:
        """
        Add a user turn directly from a FusedRepresentation.
        Salience is computed automatically.
        """
        salience = compute_salience(
            emotional_arousal=fused_result.emotional_arousal,
            fusion_confidence=fused_result.fusion_confidence,
            has_conflict=fused_result.has_semantic_conflict,
            valence=fused_result.emotional_valence,
        )
        turn = MemoryTurn(
            role="user",
            text=fused_result.text,
            emotional_label=fused_result.emotional_label,
            emotional_valence=fused_result.emotional_valence,
            emotional_arousal=fused_result.emotional_arousal,
            fusion_confidence=fused_result.fusion_confidence,
            has_conflict=fused_result.has_semantic_conflict,
            salience=salience,
            session_id=self.session_id,
        )
        self._buffer.append(turn)
        return turn

    def add_agent_turn(self, text: str) -> MemoryTurn:
        """Add an agent response (no affective metadata needed)."""
        turn = MemoryTurn(
            role="agent",
            text=text,
            session_id=self.session_id,
        )
        self._buffer.append(turn)
        return turn

    def get_turns(self) -> List[MemoryTurn]:
        """All turns in chronological order."""
        return list(self._buffer)

    def get_user_turns(self) -> List[MemoryTurn]:
        """Only user turns — used for LTM consolidation."""
        return [t for t in self._buffer if t.role == "user"]

    def get_salient_turns(self, threshold: float = 0.6) -> List[MemoryTurn]:
        """
        Turns above a salience threshold.
        These are the candidates for LTM storage at session end.
        """
        return [t for t in self._buffer if t.salience >= threshold]

    def to_langchain_messages(self) -> List[Dict]:
        """
        Format the buffer as a LangChain-compatible message list.
        Pass directly to your LLM chain as conversation history.
        """
        return [t.to_langchain_message() for t in self._buffer]

    def get_emotional_summary(self) -> Dict:
        """
        Summarise the emotional arc of the current session.

        Used to give the LLM a high-level emotional brief, e.g.:
        'The user has been mostly anxious (2 conflict moments, avg arousal 0.72)'
        """
        user_turns = self.get_user_turns()
        if not user_turns:
            return {
                "dominant_emotion": "neutral",
                "avg_valence":      0.0,
                "avg_arousal":      0.5,
                "conflict_count":   0,
                "turn_count":       0,
            }

        avg_valence    = sum(t.emotional_valence for t in user_turns) / len(user_turns)
        avg_arousal    = sum(t.emotional_arousal for t in user_turns) / len(user_turns)
        conflict_count = sum(1 for t in user_turns if t.has_conflict)
        dominant       = Counter(t.emotional_label for t in user_turns).most_common(1)[0][0]

        return {
            "dominant_emotion": dominant,
            "avg_valence":      round(avg_valence, 3),
            "avg_arousal":      round(avg_arousal, 3),
            "conflict_count":   conflict_count,
            "turn_count":       len(user_turns),
        }

    def clear(self):
        """Clear buffer, call after LTM consolidation at session end."""
        self._buffer.clear()

    def __len__(self):
        return len(self._buffer)

    def __repr__(self):
        return f"STMBuffer(session={self.session_id!r}, {len(self._buffer)}/{self.max_turns} turns)"