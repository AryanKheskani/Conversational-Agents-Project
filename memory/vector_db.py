"""
Long-Term Memory — Local FAISS Vector Store

Persists episodic memories, predictions, and distilled insights
across sessions on the user's device.

Three cognitive mechanisms (per spec):
1. Affect-Aware Retrieval: FAISS cosine search on affect-biased embeddings
2. Semantic Forgetting/Pruning: Ebbinghaus decay curve modulated by salience
3. Predict-Calibrate: Pre-date prediction → post-date insight distillation
"""

import json
import os
import time
import numpy as np
from typing import List, Tuple, Optional

import faiss

from memory.models import MemoryEntry
from memory.embedder import AffectAwareEmbedder, compute_retention


class LongTermMemory:
    """
    Local FAISS-backed long-term memory.

    Storage layout (all in `storage_path/`):
        ltm_index.faiss: FAISS flat inner-product index
        ltm_metadata.json: entry metadata (text, emotion, timestamps, etc.)

    Usage
    -----
    ltm = LongTermMemory()

    # Store a session memory:
    ltm.store_episodic("I went quiet when she asked about my job",
                        emotional_label="anxious", valence=-0.6, arousal=0.8,
                        salience=0.85, session_id="s001")

    # Retrieve by emotional + semantic similarity:
    results = ltm.retrieve("I feel nervous when asked personal questions",
                            valence=-0.5, arousal=0.75)

    # Predict-calibrate flow:
    ltm.store_prediction("User will deflect with humour under pressure", session_id="s002")
    ltm.distil_insight("User will deflect with humour under pressure",
                        "User went silent and changed the subject", session_id="s002")

    # Consolidate salient STM turns at session end:
    ltm.consolidate_from_stm(stm_buffer)
    """

    INDEX_FILE    = "ltm_index.faiss"
    METADATA_FILE = "ltm_metadata.json"

    def __init__(
        self,
        storage_path:              str   = "memory/ltm_store",
        embedder:                  Optional[AffectAwareEmbedder] = None,
        similarity_threshold:      float = 0.45,
        decay_rate:                float = 0.001,
        pruning_retention_floor:   float = 0.15,
    ):
        self.storage_path   = storage_path
        self.embedder       = embedder or AffectAwareEmbedder()
        self.sim_threshold  = similarity_threshold
        self.decay_rate     = decay_rate
        self.prune_floor    = pruning_retention_floor

        os.makedirs(storage_path, exist_ok=True)

        # FAISS inner-product index (cosine sim on L2-normalised vectors)
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.embedder.dim)
        self._entries: List[MemoryEntry] = []
        self._next_id: int = 0

        self._load()

    def store_episodic(
        self,
        text:            str,
        emotional_label: str   = "neutral",
        valence:         float = 0.0,
        arousal:         float = 0.5,
        salience:        float = 0.5,
        session_id:      str   = "",
    ) -> MemoryEntry:
        """Store a salient episodic memory from a session."""
        return self._store(text, emotional_label, valence, arousal,
                           salience, "episodic", session_id)

    def store_prediction(
        self,
        text:       str,
        session_id: str = "",
    ) -> MemoryEntry:
        """
        Store a pre-date behavioural prediction (Predict-Calibrate step 1).
        High salience by default — predictions must be revisited.
        """
        return self._store(
            text=f"[PREDICTION] {text}",
            emotional_label="neutral",
            valence=0.0, arousal=0.5,
            salience=0.9,
            entry_type="prediction",
            session_id=session_id,
        )

    def distil_insight(
        self,
        prediction_text: str,
        actual_outcome:  str,
        session_id:      str = "",
    ) -> MemoryEntry:
        """
        Predict-Calibrate distillation (Nemori/EST step 2).

        Compares what was predicted before the date against what
        actually happened. The gap is stored as a persistent insight
        that never decays — it represents hard-won self-knowledge.
        """
        insight = (
            f"[INSIGHT] "
            f"Predicted: '{prediction_text}' | "
            f"Actual: '{actual_outcome}' | "
            f"Pattern: recurring gap between rehearsed and real behaviour."
        )
        return self._store(
            text=insight,
            emotional_label="neutral",
            valence=0.0, arousal=0.5,
            salience=1.0,          # max salience, insights never decay
            entry_type="insight",
            session_id=session_id,
        )

    def consolidate_from_stm(self, stm_buffer, salience_threshold: float = 0.6) -> List[MemoryEntry]:
        """
        Consolidate salient turns from an STMBuffer into LTM.
        Call this at the end of every session.

        Only turns above `salience_threshold` are persisted:
        mundane exchanges are left to decay in STM.
        """
        salient = stm_buffer.get_salient_turns(threshold=salience_threshold)
        stored  = []

        for turn in salient:
            entry = self.store_episodic(
                text=turn.text,
                emotional_label=turn.emotional_label,
                valence=turn.emotional_valence,
                arousal=turn.emotional_arousal,
                salience=turn.salience,
                session_id=getattr(stm_buffer, "session_id", ""),
            )
            stored.append(entry)

        print(f"[LTM] Consolidated {len(stored)} turns from STM (of {len(stm_buffer.get_user_turns())} total)")
        return stored

    def retrieve(
        self,
        query_text:    str,
        valence:       float = 0.0,
        arousal:       float = 0.5,
        top_k:         int   = 5,
        min_retention: float = 0.2,
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Retrieve the most relevant memories using affect-aware cosine similarity.

        Retrieval is triggered dynamically when similarity exceeds
        `self.sim_threshold` — mirrors the spec's cosine threshold trigger.

        Returns list of (MemoryEntry, weighted_score) sorted by relevance.
        Score = cosine_similarity * retention  (faded memories rank lower)
        """
        if not self._entries:
            return []

        query_vec = self.embedder.embed(query_text, valence, arousal)
        query_vec = np.array([query_vec], dtype=np.float32)

        # Oversample then filter
        k = min(top_k * 3, len(self._entries))
        similarities, indices = self._index.search(query_vec, k)

        now     = time.time()
        results = []

        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0 or idx >= len(self._entries):
                continue
            if sim < self.sim_threshold:
                continue

            entry     = self._entries[idx]
            age       = now - entry.timestamp
            entry.retention = compute_retention(entry.salience, age, self.decay_rate)

            if entry.retention < min_retention:
                continue

            weighted = float(sim) * entry.retention
            results.append((entry, round(weighted, 4)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def prune(self) -> int:
        """
        Remove memories whose retention has decayed below the floor.
        Insights are always kept regardless of age.

        Call at session start to keep the index clean over time.
        Returns number of entries removed.
        """
        now    = time.time()
        keep   = []
        pruned = 0

        for entry in self._entries:
            age             = now - entry.timestamp
            entry.retention = compute_retention(entry.salience, age, self.decay_rate)

            if entry.entry_type == "insight":
                keep.append(entry)          # insights are permanent
            elif entry.retention >= self.prune_floor:
                keep.append(entry)
            else:
                pruned += 1

        if pruned > 0:
            print(f"[LTM] Pruned {pruned} low-retention memories")
            self._rebuild_index(keep)

        return pruned

    def _save(self):
        faiss.write_index(
            self._index,
            os.path.join(self.storage_path, self.INDEX_FILE)
        )
        with open(os.path.join(self.storage_path, self.METADATA_FILE), "w") as f:
            json.dump(
                {"entries": [e.to_dict() for e in self._entries],
                 "next_id": self._next_id},
                f, indent=2
            )

    def _load(self):
        idx_path  = os.path.join(self.storage_path, self.INDEX_FILE)
        meta_path = os.path.join(self.storage_path, self.METADATA_FILE)

        if os.path.exists(idx_path) and os.path.exists(meta_path):
            self._index   = faiss.read_index(idx_path)
            with open(meta_path) as f:
                data = json.load(f)
            self._entries = [MemoryEntry.from_dict(e) for e in data["entries"]]
            self._next_id = data.get("next_id", len(self._entries))
            print(f"[LTM] Loaded {len(self._entries)} memories from {self.storage_path}")
        else:
            print(f"[LTM] No existing memory at {self.storage_path} — starting fresh")

    def _store(
        self,
        text:            str,
        emotional_label: str,
        valence:         float,
        arousal:         float,
        salience:        float,
        entry_type:      str,
        session_id:      str,
    ) -> MemoryEntry:
        vec = self.embedder.embed(text, valence, arousal)
        self._index.add(np.array([vec], dtype=np.float32))

        entry = MemoryEntry(
            id=self._next_id,
            text=text,
            emotional_label=emotional_label,
            emotional_valence=valence,
            emotional_arousal=arousal,
            salience=salience,
            retention=1.0,
            timestamp=time.time(),
            entry_type=entry_type,
            session_id=session_id,
        )
        self._entries.append(entry)
        self._next_id += 1
        self._save()
        return entry

    def _rebuild_index(self, entries: List[MemoryEntry]):
        """Rebuild FAISS index from a filtered list after pruning."""
        self._index   = faiss.IndexFlatIP(self.embedder.dim)
        self._entries = entries

        if entries:
            vecs = self.embedder.embed_batch(
                [e.text for e in entries],
                [e.emotional_valence for e in entries],
                [e.emotional_arousal for e in entries],
            )
            self._index.add(vecs)

        self._save()

    def __len__(self):
        return len(self._entries)

    def __repr__(self):
        return f"LongTermMemory({len(self._entries)} entries @ '{self.storage_path}')"