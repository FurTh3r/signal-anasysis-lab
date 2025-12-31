import numpy as np
import soundfile as sf
import sounddevice as sd

def load_audio(file_path: str) -> tuple[np.ndarray, int]:
    x, fs = sf.read(file_path)
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    return to_mono(x), fs

def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    elif x.ndim == 2:
        mono_x = x.mean(axis=1)
        return mono_x
    else:
        raise ValueError("Input audio must be 1D or 2D array.")

def save_audio(file_path: str, x: np.ndarray, fs: int):
    sf.write(file_path, x, fs)

def reproduce_audio(x: np.ndarray, fs: int):
    sd.play(x, samplerate=fs)
    sd.wait()