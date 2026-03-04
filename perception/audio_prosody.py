# Prosodic Extraction (perception/audio_prosody.py)
import librosa
import numpy as np

def analyze_prosody(audio_path: str) -> str:
    """
    Extracts pitch (F0) and intensity (volume) from a .wav file,
    and returns a simple classified emotional tone string.
    
    Example Output: "hesitant/sad"
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Normalize the audio to make volume thresholding more robust across microphones
    if np.max(np.abs(y)) > 0:
        y = librosa.util.normalize(y)
    
    # 1. Extract Intensity (Volume) using Root Mean Square Energy
    rms = librosa.feature.rms(y=y)[0]
    avg_volume = np.mean(rms)
    
    # 2. Extract Pitch (F0) using librosa.pyin
    # fmin and fmax set to typical human voice ranges
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
    )
    
    # Filter out unvoiced segments (NaNs)
    valid_f0 = f0[~np.isnan(f0)]
    
    if len(valid_f0) > 0:
        pitch_std = np.std(valid_f0)
    else:
        pitch_std = 0.0
        
    print(f"   [Debug Tone] Volume: {avg_volume:.4f}  |  Pitch Var: {pitch_std:.4f}")
        
    # 3. Simple Classifier
    # Adjusted thresholds based on observed microphone test output
    
    is_high_volume = avg_volume >= 0.6
    is_varied_pitch = pitch_std >= 50.0
    
    if not is_high_volume and not is_varied_pitch:
        return "hesitant/sad"
    elif is_high_volume and is_varied_pitch:
        return "excited/confident"
    elif not is_high_volume and is_varied_pitch:
        return "thoughtful"
    else:
        return "assertive"

if __name__ == "__main__":
    # Optional testing logic
    print("Prosody module loaded. Run with an audio file to test.")