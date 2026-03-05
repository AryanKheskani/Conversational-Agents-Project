"""
Affect-Aware Embedder + Forgetting Curve

1. AffectAwareEmbedder
   Produces text embeddings biased toward emotional similarity.
   Inspired by MEmoRAG: emotional context is baked into the vector
   so that FAISS retrieval can match on feeling, not just topic.

2. compute_retention
   Ebbinghaus forgetting curve modulated by salience.
   High-salience memories decay much slower than mundane ones.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class AffectAwareEmbedder:
    """
    Encodes text enriched with an emotional suffix.

    Why a suffix instead of a separate vector?
    - Keeps the index simple (single FAISS index, one vector per memory)
    - The language model already understands emotional language, so
      "feeling: negative, energy: high" shifts the embedding meaningfully
    - No need for a separate acoustic feature extractor at retrieval time

    To upgrade: replace the suffix approach with true multimodal fusion
    (e.g. concatenate text embedding + [valence, arousal] features and
    reduce dimension with PCA).
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality

    def __init__(self, model_name: str = DEFAULT_MODEL):
        print(f"[Embedder] Loading sentence transformer: {model_name}")
        self.model = SentenceTransformer(model_name)
        self._dim  = self.model.get_sentence_embedding_dimension()

    def embed(
        self,
        text:     str,
        valence:  float = 0.0,
        arousal:  float = 0.5,
    ) -> np.ndarray:
        """
        Produce an L2-normalised embedding for FAISS inner-product search.
        The emotional suffix biases the vector toward emotionally similar memories.
        """
        suffix       = f"[feeling: {self._valence_word(valence)}, energy: {self._arousal_word(arousal)}]"
        enriched     = f"{text} {suffix}"
        embedding    = self.model.encode(enriched, normalize_embeddings=True)
        return embedding.astype(np.float32)

    def embed_batch(
        self,
        texts:    list,
        valences: list,
        arousals: list,
    ) -> np.ndarray:
        """Batch embed multiple entries efficiently."""
        enriched = [
            f"{t} [feeling: {self._valence_word(v)}, energy: {self._arousal_word(a)}]"
            for t, v, a in zip(texts, valences, arousals)
        ]
        embeddings = self.model.encode(enriched, normalize_embeddings=True)
        return embeddings.astype(np.float32)

    @property
    def dim(self) -> int:
        return self._dim

    @staticmethod
    def _valence_word(v: float) -> str:
        if v >  0.4: return "positive"
        if v < -0.4: return "negative"
        return "neutral"

    @staticmethod
    def _arousal_word(a: float) -> str:
        if a > 0.65: return "high"
        if a < 0.35: return "low"
        return "medium"


def compute_retention(
    salience:    float,
    age_seconds: float,
    decay_rate:  float = 0.00001,
) -> float:
    """
    Compute how much of a memory is retained after `age_seconds`.

    Formula (Ebbinghaus-inspired, modulated by salience):
        retention = salience + (1 - salience) * exp(-decay_rate * age_hours / salience)

    Effect:
    - salience=1.0 (insight/turning point) → barely decays, stays near 1.0
    - salience=0.5 (average turn)          → decays to ~0.5 over days
    - salience=0.2 (mundane turn)          → decays quickly toward pruning threshold

    Args:
        salience:    importance of the memory (0–1), set at storage time
        age_seconds: how old the memory is
        decay_rate:  controls overall decay speed (lower = slower forgetting)
    """
    if salience <= 0:
        return 0.0

    age_hours = age_seconds / 3600.0
    exponent  = -decay_rate * age_hours / max(salience, 0.05)
    retention = salience + (1.0 - salience) * np.exp(exponent)
    return float(np.clip(retention, 0.0, 1.0))