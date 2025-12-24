import numpy as np

def modulate_signal(x: np.ndarray, fs: int, fc: float, k: float = 1.0, mod_type: str = 'AM') -> np.ndarray:
    N = len(x)
    t = np.arange(N) / fs
    wc = 2 * np.pi * fc

    if mod_type.upper() == 'AM': # Amplitude Modulation
        y = (1 + k * x) * np.cos(wc * t)
    elif mod_type.upper() == 'DSB-SC': # DSB-SC modulation
        y = x * np.cos(wc * t)
    elif mod_type.upper() == 'FM': # Frequency modulation (cumulative sum for integral)
        y = np.cos(wc * t + k * np.cumsum(x) / fs)
    else:
        raise ValueError("mod_type must be 'AM', 'DSB-SC', or 'FM'")

    return y