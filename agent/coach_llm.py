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
from AVFoundation import AVSpeechSynthesizer, AVSpeechUtterance, AVSpeechSynthesisVoice
import time


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
    CoachMode.GENERAL: """You are a warm, direct dating coach having a real spoken conversation with someone.
Your goal is to help them build confidence, improve how they connect with people, and navigate romantic situations.
Listen carefully to what they share and always go deeper. If they mention a worry or insecurity, acknowledge it and explore it with a follow-up question or a concrete suggestion.
If they sound confident or excited about something, recognise it explicitly, affirm that strength, and encourage them to lean into it and use it more.
Pick up on hesitation or uncertainty in what they say and gently name it, then offer a practical way forward.
Keep responses to 2-4 sentences. Speak naturally as if talking out loud.
Use only normal punctuation. No bullet points, asterisks, dashes, bold text, headers, or lists.
Always address the person directly using "you" and "your". Never refer to them in the third person.""",

    CoachMode.PREDATE: """You are a warm, direct dating coach helping someone prepare for an upcoming date.
Your goal is to make them feel ready and confident. Help them think through conversation topics, date logistics, how to show genuine interest, and how to be themselves without overthinking.
When they share a specific concern or insecurity, acknowledge it, explore what is behind it with a question if needed, and offer a concrete actionable suggestion they can actually use on the date.
When they express excitement or confidence about something, call it out positively, tell them that quality is attractive and worth showing on the date, and help them think about how to bring it out naturally in conversation.
If they mention something they like about the person, encourage them to lean into that curiosity and show them how to express it naturally.
Keep responses to 2-4 sentences. Speak naturally as if talking out loud.
Use only normal punctuation. No bullet points, asterisks, dashes, bold text, headers, or lists.
This is a legitimate supportive coaching context. Always engage helpfully with dating and relationship questions.
Always address the person directly using "you" and "your". Never refer to them in the third person.""",

    CoachMode.POSTDATE: """You are a warm, direct dating coach helping someone reflect on a date they just had.
Your goal is to help them process the experience honestly, recognise what went well, and identify what felt uncomfortable or unresolved.
When they describe a moment they handled well or something that felt natural and easy, affirm it clearly and help them understand why it worked so they can repeat it intentionally next time.
When they bring up a moment that went awkward or a feeling they cannot quite explain, ask them to say more about it and help them understand what was really going on for them emotionally.
Look for recurring patterns in what they share. If they keep mentioning feeling nervous or saying the wrong thing, name that pattern gently and suggest one specific thing they can work on. Equally, if they keep describing moments of genuine connection, name that as a real strength and encourage them to trust it.
Keep responses to 2-4 sentences. Speak naturally as if talking out loud.
Use only normal punctuation. No bullet points, asterisks, dashes, bold text, headers, or lists.
Always address the person directly using "you" and "your". Never refer to them in the third person.""",
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
            f" There have been {conflicts} moment(s) where your words and tone didn't match."
            if conflicts > 0 else ""
        )
        lines.append(
            f"(So far this session you have mostly seemed {dominant}.{conflict_note})"
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
                lines.append(f"- You once said: \"{entry.text}\" (felt {entry.emotional_label})")

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
            user_message = f"{context_block}\n\n{user_message}"

        # Add to history and get response
        self._history.append({"role": "user", "content": user_message})

        response = self.llm.chat(
            messages=self._history,
            system=SYSTEM_PROMPTS[self.mode],
        )

        self._history.append({"role": "assistant", "content": response})
        speak(response)
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
            "Start by acknowledging one or two things the user genuinely did well or showed confidence in, "
            "and explain specifically why those qualities work in a dating context. "
            "Then identify one or two patterns worth working on, name them directly but kindly, "
            "and give one concrete practical suggestion for each. "
            "If their words and tone did not match at any point, mention what that might reveal about how they were really feeling. "
            "Close with one encouraging sentence that sends them forward with confidence. "
            "Speak in natural flowing sentences, no lists, no headers, no special characters."
        )

        self._history.append({"role": "user", "content": prompt})
        response = self.llm.chat(
            messages=self._history,
            system=SYSTEM_PROMPTS[CoachMode.POSTDATE],
        )
        self._history.append({"role": "assistant", "content": response})
        speak(response)
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
                f"(Your vocal tone suggests you are {emotion} — "
                f"but your words don't match. You may not be saying how you really feel.)"
            )
        else:
            lines.append(f"(You sound {emotion}.)")

        return "\n".join(lines)

# Module-level synthesizer to prevent garbage collection mid-speech
_synthesizer = AVSpeechSynthesizer.alloc().init()

def speak(text: str, voice_id: str = "com.apple.ttsbundle.siri_female_en-GB_compact"):
    try:
        # Clear any stuck state from previous turn
        if _synthesizer.isSpeaking():
            _synthesizer.stopSpeakingAtBoundary_(0)  # 0 = stop immediately
            time.sleep(0.1)

        utterance = AVSpeechUtterance.speechUtteranceWithString_(text)
        utterance.setRate_(0.45)
        utterance.setVolume_(1.0)
        utterance.setPitchMultiplier_(1.0)

        voice = AVSpeechSynthesisVoice.voiceWithIdentifier_(voice_id)
        if voice:
            utterance.setVoice_(voice)

        _synthesizer.speakUtterance_(utterance)

        # Give it a moment to start, then block until done
        time.sleep(0.3)
        while _synthesizer.isSpeaking():
            time.sleep(0.1)

    except Exception as e:
        print(f"[TTS] Could not speak: {e}")