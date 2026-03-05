"""
Modality Fusion Engine

Fuses ASR transcript (from audio_asr.py) with prosodic tone
(from audio_prosody.py) to produce a unified emotional representation.

Detects alignment or conflict between what is said and how it is said.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from transformers import pipeline
import re


# Configuration: all tuneable values live here

@dataclass
class ProsodyConfig:
    """
    Maps prosody classifier labels → (valence, arousal) in [-1,1] x [0,1].
    Swap out or extend this dict to match whatever labels the classifier emits.
    """
    label_map: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "excited/confident": ( 0.8, 0.9),
        "assertive":         ( -0.1, 0.8),
        "thoughtful":        ( 0.45, 0.4),
        "hesitant/sad":      (-0.5, 0.3),
    })
    fallback: Tuple[float, float] = (0.0, 0.5)

    # Per-label reliability: distinctive patterns are more trustworthy
    reliability_map: Dict[str, float] = field(default_factory=lambda: {
        "excited/confident": 0.85,
        "assertive":         0.75,
        "thoughtful":        0.75,
        "hesitant/sad":      0.85,
    })
    fallback_reliability: float = 0.70


@dataclass
class FusionConfig:
    """Weights and thresholds that control fusion behaviour."""
    # How much to trust prosody vs semantics when there is a conflict
    conflict_prosody_weight: float = 0.70
    conflict_semantic_weight: float = 0.30

    # Minimum valence gap to call something a conflict
    conflict_valence_threshold: float = 0.60

    # Confidence penalty when modalities conflict
    conflict_confidence_penalty: float = 0.85

    # Known linguistic conflict patterns: (text_keywords, prosody_label)
    conflict_patterns: List[Tuple[List[str], str]] = field(default_factory=lambda: [
        (["fine", "good", "okay", "great"], "hesitant/sad"),     # "I'm fine" + sad tone
        (["bad", "terrible", "awful", "hate"], "excited/confident"),  # negatives + happy tone
    ])


# Semantic Analyser

class SemanticAnalyzer:
    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None  # return all emotion scores
        )

    # Emotions this model outputs that correspond to high arousal
    HIGH_AROUSAL_EMOTIONS = {"anger", "fear", "surprise", "disgust"}
    LOW_AROUSAL_EMOTIONS  = {"sadness", "neutral"}
    POSITIVE_EMOTIONS     = {"joy"}
    NEGATIVE_EMOTIONS     = {"anger", "fear", "sadness", "disgust"}

    def _has_negation(self, transcript: str) -> bool:
        """Detects negation using common patterns."""
        negation_emotion_pattern = re.compile(
            r"\b(not|never|don't|doesn't|didn't|won't|can't|isn't|aren't)\b"
            r".{0,20}"  # within ~4 words
            r"\b(nervous|scared|anxious|worried|happy|excited|sad|angry|fine|good|okay|great)\b",
            re.IGNORECASE
        )
        return bool(negation_emotion_pattern.search(transcript))

    def analyze(self, transcript: str) -> Tuple[float, float, float]:
        if not transcript or not transcript.strip():
            return 0.0, 0.5, 0.0

        results = self.classifier(transcript)[0]  # list of {label, score}
        scores  = {r["label"].lower(): r["score"] for r in results}

        # Valence: positive emotions push up, negative push down
        valence = (
                sum(scores.get(e, 0) for e in self.POSITIVE_EMOTIONS) -
                sum(scores.get(e, 0) for e in self.NEGATIVE_EMOTIONS)
        )

        # Flip valence if negation is detected
        if self._has_negation(transcript):
            valence = valence * -0.8  # flip but dampen — "not great" ≠ "terrible"

        # Arousal: high-energy emotions push up, low-energy push down
        arousal = 0.5 + (
                sum(scores.get(e, 0) for e in self.HIGH_AROUSAL_EMOTIONS) -
                sum(scores.get(e, 0) for e in self.LOW_AROUSAL_EMOTIONS)
        ) * 0.5

        # Reliability: confidence of top prediction
        top_score   = max(scores.values())
        reliability = 0.5 + top_score * 0.5

        # Pass dominant emotion through for better labelling
        self.last_dominant = max(scores, key=scores.get)

        return (
            float(np.clip(valence, -1.0, 1.0)),
            float(np.clip(arousal, 0.0, 1.0)),
            float(np.clip(reliability, 0.0, 1.0)),
        )


# Conflict Detector

class ConflictDetector:
    """
    Decides whether text content and vocal tone are misaligned.
    Configurable via FusionConfig.
    """

    def __init__(self, config: FusionConfig):
        self.config = config

    def has_conflict(self, semantic_valence, prosodic_valence,
                     transcript, prosody_label,
                     semantic_arousal=0.5, prosodic_arousal=0.5) -> bool:

        # Thoughtful is valence-neutral, never treat it as conflicting
        if prosody_label == "thoughtful":
            return False

        # If both modalities agree on high arousal, don't call it a conflict
        # even if valence differs
        both_high_arousal = semantic_arousal > 0.7 and prosodic_arousal > 0.6
        if both_high_arousal:
            return False

        if abs(semantic_valence - prosodic_valence) > self.config.conflict_valence_threshold:
            return True

        text_lower = transcript.lower()
        for keywords, conflicting_label in self.config.conflict_patterns:
            if prosody_label == conflicting_label and any(kw in text_lower for kw in keywords):
                return True

        return False


# Emotion label mapping

def map_to_label(valence: float, arousal: float, dominant_emotion: str = "") -> str:
    # Trust the model's direct emotion label when confident
    direct_map = {
        "joy":      "excited" if arousal > 0.6 else "content",
        "anger":    "angry",
        "fear":     "anxious",
        "sadness":  "sad",
        "disgust":  "frustrated",
        "surprise": "surprised",
        "neutral":  "neutral",
    }
    if dominant_emotion in direct_map:
        return direct_map[dominant_emotion]

    # Fallback to valence/arousal mapping
    if valence > 0.3:
        return "excited" if arousal > 0.7 else "content"
    elif valence < -0.3:
        return "angry" if arousal > 0.6 else "sad"
    else:
        return "aroused" if arousal > 0.7 else "calm" if arousal < 0.3 else "neutral"


# Output dataclass

@dataclass
class FusedRepresentation:
    """Unified output ready for memory storage or downstream agents."""
    text: str
    emotional_label: str        = "neutral"
    emotional_valence: float    = 0.0   # -1 (negative) → +1 (positive)
    emotional_arousal: float    = 0.5   # 0 (calm)      →  1 (activated)

    semantic_reliability: float  = 1.0
    prosodic_reliability: float  = 1.0
    fusion_confidence: float     = 1.0

    has_semantic_conflict: bool  = False
    dominant_modality: str       = "semantic"

    # Human-readable conflict explanation
    conflict_note: str           = ""

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


# Main engine

class ModalityFusionEngine:
    """
    Fuses ASR transcript with prosodic tone classification.

    Usage
    -----
    # From pre-computed values:
    engine = ModalityFusionEngine()
    result = engine.fuse(transcript="I'm fine", prosody_label="hesitant/sad")

    # Directly from a .wav file (calls audio_asr + audio_prosody internally):
    result = engine.fuse_from_audio("path/to/recording.wav")
    """

    def __init__(
            self,
            prosody_config: Optional[ProsodyConfig]  = None,
            fusion_config:  Optional[FusionConfig]   = None,
            semantic_analyzer: Optional[SemanticAnalyzer] = None,
    ):
        self.prosody_cfg  = prosody_config   or ProsodyConfig()
        self.fusion_cfg   = fusion_config    or FusionConfig()
        self.analyzer     = semantic_analyzer or SemanticAnalyzer()
        self.conflict_det = ConflictDetector(self.fusion_cfg)

    # Public API
    def fuse(self, transcript: str, prosody_label: str) -> FusedRepresentation:
        """
        Fuse pre-computed transcript + prosody label into a unified representation.
        """
        # Semantic signal
        sem_valence, sem_arousal, sem_reliability = self.analyzer.analyze(transcript)

        # Prosodic signal
        pro_valence, pro_arousal = self.prosody_cfg.label_map.get(
            prosody_label, self.prosody_cfg.fallback
        )
        pro_reliability = self.prosody_cfg.reliability_map.get(
            prosody_label, self.prosody_cfg.fallback_reliability
        )

        # Conflict detection
        conflict = self.conflict_det.has_conflict(
            sem_valence, pro_valence, transcript, prosody_label,
            semantic_arousal=sem_arousal, prosodic_arousal=pro_arousal
        )

        # Compute fusion weights
        if conflict:
            w_sem = self.fusion_cfg.conflict_semantic_weight
            w_pro = self.fusion_cfg.conflict_prosody_weight
            dominant = "prosodic"
            conflict_note = (
                f"Valence gap {abs(sem_valence - pro_valence):.2f} — "
                f"trusting prosody (vocal tone harder to fake)"
            )
        else:
            total = sem_reliability + pro_reliability
            w_sem = sem_reliability / total if total > 0 else 0.5
            w_pro = pro_reliability / total if total > 0 else 0.5
            dominant = "prosodic" if pro_reliability > sem_reliability else "semantic"
            conflict_note = ""

        # Weighted fusion
        fused_valence = np.clip(w_sem * sem_valence + w_pro * pro_valence, -1.0, 1.0)
        fused_arousal = np.clip(w_sem * sem_arousal + w_pro * pro_arousal,  0.0, 1.0)

        # Confidence
        fusion_confidence = (sem_reliability + pro_reliability) / 2
        if conflict:
            fusion_confidence *= self.fusion_cfg.conflict_confidence_penalty

        dominant_emotion = getattr(self.analyzer, "last_dominant", "")

        return FusedRepresentation(
            text=transcript,
            emotional_label = map_to_label(fused_valence, fused_arousal, dominant_emotion),
            emotional_valence=round(float(fused_valence), 3),
            emotional_arousal=round(float(fused_arousal), 3),
            semantic_reliability=round(sem_reliability, 3),
            prosodic_reliability=round(pro_reliability, 3),
            fusion_confidence=round(fusion_confidence, 3),
            has_semantic_conflict=conflict,
            dominant_modality=dominant,
            conflict_note=conflict_note,
        )

    def fuse_from_audio(self, audio_path: str) -> FusedRepresentation:
        """
        Full pipeline: .wav → transcript + prosody → FusedRepresentation.
        Imports audio_asr and audio_prosody lazily so the engine stays
        usable without them in unit-test environments.
        """
        try:
            from perception.audio_asr import transcribe_audio
            from perception.audio_prosody import analyze_prosody
        except ImportError as e:
            raise ImportError(
                "Could not import perception modules. "
                "Make sure audio_asr.py and audio_prosody.py are on your path."
            ) from e

        transcript    = transcribe_audio(audio_path)
        prosody_label = analyze_prosody(audio_path)
        return self.fuse(transcript, prosody_label)


# CLI test

if __name__ == "__main__":
    engine = ModalityFusionEngine()

    test_cases = [
        # (transcript,                    prosody_label,        expected_note)
        ("I'm really excited about tonight!", "excited/confident", "aligned — both positive"),
        ("Yeah, I'm fine with whatever",      "hesitant/sad",      "CONFLICT — trust prosody"),
        ("The date was terrible",             "hesitant/sad",      "aligned — both negative"),
        ("I'm not nervous at all",            "hesitant/sad",      "CONFLICT — trust prosody"),
        ("I had a good time",                 "thoughtful",        "mild positive, low arousal"),
    ]

    print("=" * 60)
    print("Modality Fusion Engine — smoke test")
    print("=" * 60)

    for transcript, prosody, note in test_cases:
        r = engine.fuse(transcript, prosody)
        print(f"\n  Text   : {transcript!r}")
        print(f"  Prosody: {prosody}")
        print(f"  Expect : {note}")
        print(f"  → Label: {r.emotional_label}  "
              f"(V={r.emotional_valence:+.2f}, A={r.emotional_arousal:.2f})")
        print(f"    Conflict : {r.has_semantic_conflict}  |  Dominant: {r.dominant_modality}")
        if r.conflict_note:
            print(f"    Note     : {r.conflict_note}")
        print(f"    Confidence: {r.fusion_confidence:.2f}  "
              f"(sem={r.semantic_reliability:.2f}, pro={r.prosodic_reliability:.2f})")
