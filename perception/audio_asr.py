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

def record_audio(filename="temp_recording.wav"):
    """
    Push-to-talk recording: hold SPACE to record, release to stop.
    Records until the user releases the spacebar.
    Records audio from the microphone for a specified duration and saves it to a .wav file.
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    press_event = threading.Event()
    release_event = threading.Event()
    frames = []
    sample_width = [0]  # mutable container to pass value out of thread

    def on_press(key):
        if key == keyboard.Key.space:
            press_event.set()

    def on_release(key):
        if key == keyboard.Key.space:
            release_event.set()
            return False

    def record_thread():
        p = pyaudio.PyAudio()
        sample_width[0] = p.get_sample_size(FORMAT)  # capture before terminate
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)

        # Wait for spacebar before capturing frames
        press_event.wait()
        print("🔴 Recording...")

        # Drain buffered audio from before keypress
        stream.read(stream.get_read_available(), exception_on_overflow=False)

        while not release_event.is_set():
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

    print("\n🎙️  Hold SPACE to speak, release to stop...")

    # Start recording thread first
    t = threading.Thread(target=record_thread, daemon=True)
    t.start()

    # Keyboard listener runs on main thread — not blocked by audio I/O
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    # Wait for recording thread to finish cleanly
    t.join()

    print("✅ Finished recording.")

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