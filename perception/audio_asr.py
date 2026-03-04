# Semantic Extraction (perception/audio_asr.py)

import whisper
import warnings
import pyaudio
import wave
import tempfile
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

def record_audio(filename="temp_recording.wav", record_seconds=5):
    """
    Records audio from the microphone for a specified duration and saves it to a .wav file.
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()

    print(f"\n🎙️ Listening for {record_seconds} seconds...")

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    try:
        for _ in range(0, int(RATE / CHUNK * record_seconds)):
            data = stream.read(CHUNK)
            frames.append(data)
    except KeyboardInterrupt:
        pass

    print("✅ Finished recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
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