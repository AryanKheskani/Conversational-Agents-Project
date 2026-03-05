"""
Shared data models for the memory layer.
Both STM and LTM import from here, no circular dependencies.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class MemoryTurn:
    """
    A single episodic unit, one thing the user said, enriched with
    affective metadata from the fusion engine.

    This is the atomic unit that flows from STM → LTM.
    """
    role: str                          # "user" | "agent"
    text: str                          # transcript or agent response
    timestamp: float = field(default_factory=time.time)

    # Affective metadata (populated from FusedRepresentation)
    emotional_label:   str   = "neutral"
    emotional_valence: float = 0.0     # -1 (negative) → +1 (positive)
    emotional_arousal: float = 0.5     # 0 (calm)      →  1 (activated)
    fusion_confidence: float = 1.0
    has_conflict:      bool  = False   # prosody/semantic mismatch detected

    # Salience: how memorable this turn is (drives LTM retention)
    salience: float = 0.5

    # Session this turn belongs to
    session_id: str = ""

    def to_dict(self) -> Dict:
        return self.__dict__.copy()

    def to_langchain_message(self) -> Dict:
        """Format for LangChain message history / LLM context window."""
        return {
            "role":    self.role,
            "content": self.text,
            "metadata": {
                "emotional_label":   self.emotional_label,
                "emotional_valence": self.emotional_valence,
                "emotional_arousal": self.emotional_arousal,
                "fusion_confidence": self.fusion_confidence,
                "has_conflict":      self.has_conflict,
                "salience":          self.salience,
                "session_id":        self.session_id,
            }
        }


@dataclass
class MemoryEntry:
    """
    A persisted long-term memory unit.
    Can be an episodic summary, a behavioural prediction, or a
    distilled insight about the user's patterns.
    """
    id:        int
    text:      str
    timestamp: float = field(default_factory=time.time)

    # Affective metadata
    emotional_label:   str   = "neutral"
    emotional_valence: float = 0.0
    emotional_arousal: float = 0.5

    # Retention mechanics
    salience:  float = 0.5    # initial importance (does not change)
    retention: float = 1.0    # decays over time via forgetting curve

    # Entry classification
    entry_type: str = "episodic"   # "episodic" | "prediction" | "insight"
    session_id: str = ""

    def to_dict(self) -> Dict:
        return self.__dict__.copy()

    @staticmethod
    def from_dict(d: Dict) -> "MemoryEntry":
        return MemoryEntry(**d)


def compute_salience(
    emotional_arousal: float,
    fusion_confidence: float,
    has_conflict:      bool,
    valence:           float,
) -> float:
    """
    How memorable is this moment?

    Weighting rationale:
    - Arousal is the strongest predictor of memory consolidation
    - Extreme valence (very positive or very negative) enhances recall
    - Conflict (saying one thing, feeling another) is always notable
    - Higher confidence in the reading = more reliable signal
    """
    score  = emotional_arousal * 0.50
    score += abs(valence)      * 0.20
    score += 0.20 if has_conflict else 0.0
    score += fusion_confidence * 0.10
    return round(min(score, 1.0), 3)