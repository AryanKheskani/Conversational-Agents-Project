import json
import os
from datetime import datetime


class SessionLogger:
    def __init__(self, session_id: str, log_dir: str = "memory/session_logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.session_id = session_id
        self.path = os.path.join(log_dir, f"{session_id}.json")
        self._data = {
            "session_id": session_id,
            "started_at": datetime.now().isoformat(),
            "ended_at":   None,
            "mode":       None,
            "use_prosody": None,
            "turns":      [],
        }
        self._save()

    def log_user_turn(self, result, salience: float, tone: str = None):
        self._data["turns"].append({
            "role":          "user",
            "timestamp":     datetime.now().isoformat(),
            "text":          result.text,
            "tone":          tone,
            "emotion":       result.emotional_label,
            "valence":       result.emotional_valence,
            "arousal":       result.emotional_arousal,
            "confidence":    result.fusion_confidence,
            "salience":      salience,
            "conflict":      result.has_semantic_conflict,
            "conflict_note": result.conflict_note,
        })
        self._save()

    def log_agent_turn(self, text: str):
        self._data["turns"].append({
            "role":      "agent",
            "timestamp": datetime.now().isoformat(),
            "text":      text,
        })
        self._save()

    def log_retrieved_memories(self, memories: list):
        """Attach retrieved memories to the last user turn."""
        for turn in reversed(self._data["turns"]):
            if turn["role"] == "user":
                turn["retrieved_memories"] = memories
                break
        self._save()

    def log_reflection(self, text: str):
        self._data["reflection"] = text
        self._save()

    def close(self, summary: dict):
        self._data["ended_at"] = datetime.now().isoformat()
        self._data["summary"]  = summary
        self._save()
        print(f"[Logger] Session saved to {self.path}")

    def set_meta(self, mode: str, use_prosody: bool):
        self._data["mode"]        = mode
        self._data["use_prosody"] = use_prosody
        self._save()

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)