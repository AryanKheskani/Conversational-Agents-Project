# Semantic Extraction (perception/audio_asr.py)

import whisper
import warnings
import pyaudio
import wave
import tempfile
import threading
from pynput import keyboard
import os

# Suppress FP16 warnings on CPU
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Cache the model to avoid reloading it on every function call
_model = None

def transcribe_audio(audio_path: str) -> str:
    """
    Takes a .wav file, runs it through whisper.load_model("base"), 
    and returns the transcribed string.
    
    Example Output: "I guess the date was okay."
    """
    global _model
    if _model is None:
        _model = whisper.load_model("base")
        
    result = _model.transcribe(audio_path)
    return result["text"].strip()

def record_audio(filename="temp_recording.wav", external_stop: threading.Event = None, auto_start: bool = False):
    CHUNK    = 1024
    FORMAT   = pyaudio.paInt16
    CHANNELS = 1
    RATE     = 16000

    press_event   = threading.Event()
    release_event = threading.Event()
    quit_event    = threading.Event()
    frames        = []
    sample_width  = [0]

    # Auto-start from browser — no keypress needed
    if auto_start:
        press_event.set()

    def on_press(key):
        if key == keyboard.Key.space:
            press_event.set()
        elif hasattr(key, 'char') and key.char == 's':
            if not press_event.is_set():
                press_event.set()
            else:
                release_event.set()
                return False
        elif hasattr(key, 'char') and key.char == 'q':
            quit_event.set()
            press_event.set()
            release_event.set()
            return False

    def on_release(key):
        if key == keyboard.Key.space:
            release_event.set()
            return False

    def record_thread():
        p = pyaudio.PyAudio()
        sample_width[0] = p.get_sample_size(FORMAT)
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)

        press_event.wait()

        if not quit_event.is_set():
            print("🔴 Recording...")
            stream.read(stream.get_read_available(), exception_on_overflow=False)
            while not release_event.is_set():
                if external_stop and external_stop.is_set():
                    release_event.set()
                    break
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

    if not auto_start:
        print("\n🎙️  Hold SPACE to speak, release to stop  |  S to toggle  |  Q to quit...")

    t = threading.Thread(target=record_thread, daemon=True)
    t.start()

    if auto_start:
        # Browser mode — no keyboard listener needed, just wait for external_stop
        release_event.wait()
        t.join()
    else:
        # Terminal mode — keyboard listener drives start/stop
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
        t.join()

    if quit_event.is_set():
        return None

    print("✅ Done.")

    if not frames:
        return None

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(sample_width[0])
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    return filename

if __name__ == "__main__":
    print("Initializing ASR system...")
    try:
        while True:
            audio_file = record_audio(record_seconds=5)
            print("⏳ Transcribing...")
            text = transcribe_audio(audio_file)
            print(f"🗣️ You said: {text}")
            
            # Clean up the temporary file
            if os.path.exists(audio_file):
                os.remove(audio_file)
                
    except KeyboardInterrupt:
        print("\nStopping ASR system...")