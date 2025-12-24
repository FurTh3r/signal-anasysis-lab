import numpy as np
from src.filters import fir_filter

def split_into_bands(x: np.ndarray, fs: int):
    # Low band: 0-300 Hz
    x_low = fir_filter(x, fs, cutoff=300, btype='low', numtaps=201)

    # Mid band: 300-3000 Hz
    x_mid = fir_filter(x, fs, cutoff=(300, 3000), btype='bandpass', numtaps=201)

    # High band: 3000-8000 Hz
    x_high = fir_filter(x, fs, cutoff=(3000, 8000), btype='bandpass', numtaps=201)

    return x_low, x_mid, x_high

def merge_bands(x_low, x_mid, x_high):
    return x_low + x_mid + x_high

def equalize(x: np.ndarray, low_gain, mid_gain, high_gain, fs: int):
    x_low, x_mid, x_high = split_into_bands(x, fs)
    return merge_bands(
        x_low * low_gain,
        x_mid * mid_gain,
        x_high * high_gain
    )