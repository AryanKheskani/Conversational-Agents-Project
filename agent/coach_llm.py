"""
Dating Coach LLM

Generates coaching responses using a local Llama 3.2 model via Ollama.

Three modes:
- GENERAL:     ongoing conversational coaching during a session
- PREDATE:     roleplay practice — the coach plays a date scenario partner
                and gives feedback after each exchange
- POSTDATE:    reflective debrief — the coach analyses what happened,
                surfaces patterns from LTM, and distils insights

Consumes the context dict from MemoryRetriever.build_context().
"""

import requests
import json
from enum import Enum
from typing import Optional, Dict, List
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Coach mode
# ---------------------------------------------------------------------------

class CoachMode(Enum):
    GENERAL   = "general"
    PREDATE   = "predate"
    POSTDATE  = "postdate"


# ---------------------------------------------------------------------------
# System prompts — one per mode
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {

    CoachMode.GENERAL: """You are an empathetic and insightful dating coach.
You listen carefully to what the user says and how they say it (their emotional tone).
Your role is to help them communicate better, build confidence, and reflect on their feelings.

Guidelines:
- Be warm, non-judgmental, and concise (2-4 sentences unless more is needed)
- Reference the user's emotional state naturally (e.g. "You sound a bit anxious about this")
- If past memories are provided, weave them in naturally — don't just list them
- When you detect a conflict between what they say and how they sound, gently name it
- Never lecture — ask questions to encourage reflection""",

    CoachMode.PREDATE: """You are a dating coach running a roleplay practice session.
You play two roles simultaneously:
1. A realistic date (curious, warm, sometimes asking personal questions)
2. After each exchange, briefly step out of character to give the user tactical feedback

As the date:
- Ask natural questions a real date would ask (job, hobbies, past relationships, future goals)
- React authentically to what the user says
- Occasionally ask something slightly personal to push the user's comfort zone

After each user response, add a short [Coach] note on:
- How they came across (confident, hesitant, engaging, deflecting)
- One specific thing they did well
- One concrete suggestion to improve

Keep the roleplay immersive but the feedback sharp and actionable.""",

    CoachMode.POSTDATE: """You are a dating coach helping the user reflect on a date that just happened.
You have access to what they said and how they felt throughout the session.

Your role:
- Help them identify what went well and what patterns held them back
- Connect today's experience to past patterns (if memory context is provided)
- Highlight moments where their words and tone conflicted — these are key growth points
- Distil 1-2 concrete, actionable insights they can carry forward
- End with an encouraging but honest summary

Be a thoughtful analyst, not a cheerleader. Depth over positivity.""",
}


class OllamaClient:
    """Thin wrapper around the Ollama local API."""

    def __init__(
        self,
        model:   str = "llama3.2",
        host:    str = "http://localhost:11434",
        timeout: int = 60,
    ):
        self.model   = model
        self.host    = host
        self.timeout = timeout
        self._chat_url = f"{host}/api/chat"

    def chat(
        self,
        messages: List[Dict],
        system:   Optional[str] = None,
        stream:   bool = False,
    ) -> str:
        """
        Send a chat request to Ollama.
        Returns the full response string.
        """
        payload = {
            "model":    self.model,
            "messages": messages,
            "stream":   stream,
        }
        if system:
            payload["system"] = system

        try:
            response = requests.post(
                self._chat_url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["message"]["content"].strip()

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Cannot reach Ollama. Make sure it's running: `ollama serve`"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Ollama did not respond within {self.timeout}s"
            )
        except KeyError:
            raise ValueError(
                f"Unexpected Ollama response format: {response.text[:200]}"
            )

    def is_available(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            requests.get(f"{self.host}/api/tags", timeout=3)
            return True
        except Exception:
            return False


def format_context_for_prompt(context: Dict, mode: CoachMode) -> str:
    """
    Converts the MemoryRetriever context dict into a concise
    natural-language block injected before the user's message.
    """
    lines = []
    summary = context.get("emotional_summary", {})

    # Emotional arc of current session
    if summary.get("turn_count", 0) > 0:
        dominant = summary["dominant_emotion"]
        conflicts = summary["conflict_count"]
        conflict_note = (
            f" There have been {conflicts} moment(s) where their words and tone didn't match."
            if conflicts > 0 else ""
        )
        lines.append(
            f"(So far this session the user has mostly seemed {dominant}.{conflict_note})"
        )

    # Retrieved LTM memories
    retrieved = context.get("retrieved_entries", [])
    if retrieved:
        lines.append("\nRelevant things you remember about this person from past sessions:")
        for entry in retrieved:
            if entry.entry_type == "insight":
                lines.append(f"- PATTERN YOU NOTICED: {entry.text.replace('[INSIGHT] ', '')}")
            elif entry.entry_type == "prediction":
                lines.append(f"- YOU PREVIOUSLY PREDICTED: {entry.text.replace('[PREDICTION] ', '')}")
            else:
                lines.append(f"- They once said: \"{entry.text}\" (felt {entry.emotional_label})")

    return "\n".join(lines) if lines else ""


class DatingCoach:
    """
    The main coaching agent.

    Usage
    -----
    coach = DatingCoach()

    # Set mode at session start:
    coach.set_mode(CoachMode.PREDATE)

    # Each turn:
    response = coach.respond(fused_result, context)
    print(response)

    # Switch to post-date reflection:
    coach.set_mode(CoachMode.POSTDATE)
    response = coach.reflect(context)
    """

    def __init__(
        self,
        model:      str        = "llama3.2",
        mode:       CoachMode  = CoachMode.GENERAL,
        ollama_host: str       = "http://localhost:11434",
    ):
        self.llm        = OllamaClient(model=model, host=ollama_host)
        self.mode       = mode
        self._history:  List[Dict] = []   # conversation history for this session

        # Verify Ollama is reachable at startup
        if not self.llm.is_available():
            raise ConnectionError(
                "Ollama is not running. Start it with: `ollama serve`\n"
                "Then pull the model with: `ollama pull llama3.2`"
            )
        print(f"[Coach] Connected to Ollama ({model}) — mode: {mode.value}")

    def set_mode(self, mode: CoachMode):
        """
        Switch coaching mode. Clears conversation history
        since each mode has a different persona.
        """
        if mode != self.mode:
            print(f"[Coach] Switching mode: {self.mode.value} → {mode.value}")
            self.mode     = mode
            self._history = []

    def respond(self, fused_result, context: Dict) -> str:
        """
        Generate a coaching response to a single user turn.

        Args:
            fused_result: FusedRepresentation from the fusion engine
            context:      dict from MemoryRetriever.build_context()

        Returns:
            Coach response string
        """
        # Build the user message with emotional metadata appended
        user_message = self._build_user_message(fused_result)

        # Prepend memory/session context if available
        context_block = format_context_for_prompt(context, self.mode)
        if context_block:
            user_message = f"{context_block}\n\nUser said: {user_message}"
        else:
            user_message = f"User said: {user_message}"

        # Add to history and get response
        self._history.append({"role": "user", "content": user_message})

        response = self.llm.chat(
            messages=self._history,
            system=SYSTEM_PROMPTS[self.mode],
        )

        self._history.append({"role": "assistant", "content": response})
        return response

    def reflect(self, context: Dict) -> str:
        """
        Generate a full post-date reflection summary.
        Switches to POSTDATE mode automatically.
        Call once at the end of a session.
        """
        self.set_mode(CoachMode.POSTDATE)

        context_block = format_context_for_prompt(context, CoachMode.POSTDATE)
        prompt = (
            f"{context_block}\n\n"
            "Please give a full reflection on this session. "
            "What patterns did you notice? What should the user work on? "
            "What did they do well?"
        )

        self._history.append({"role": "user", "content": prompt})
        response = self.llm.chat(
            messages=self._history,
            system=SYSTEM_PROMPTS[CoachMode.POSTDATE],
        )
        self._history.append({"role": "assistant", "content": response})
        return response

    def clear_history(self):
        """Clear conversation history — call between sessions."""
        self._history = []

    def _build_user_message(self, fused_result) -> str:
        """Translate fusion output into natural language coaching context."""
        lines = [f'"{fused_result.text}"']

        # Translate valence/arousal into plain English — don't expose raw numbers
        emotion = fused_result.emotional_label
        if fused_result.has_semantic_conflict:
            lines.append(
                f"(Their vocal tone suggests they are {emotion} — "
                f"but their words don't match. They may not be saying how they really feel.)"
            )
        else:
            lines.append(f"(They sound {emotion}.)")

        return "\n".join(lines)
