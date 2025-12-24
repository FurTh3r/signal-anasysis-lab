import numpy as np
from scipy.signal import firwin, lfilter, freqz

def fir_filter(x: np.ndarray, fs: int, cutoff, btype: str = 'low', numtaps: int = 101):
    nyq = 0.5 * fs # Nysquist frequency

    # Design FIR coefficients
    if btype == 'low':
        b = firwin(numtaps, cutoff/nyq)
    elif btype == 'high':
        b = firwin(numtaps, cutoff/nyq, pass_zero=False)
    elif btype == 'bandpass':
        b = firwin(numtaps, [c/nyq for c in cutoff], pass_zero=False)
    else:
        raise ValueError("btype must be 'low', 'high', or 'bandpass'")

    # Apply filter
    y = lfilter(b, 1.0, x)
    return y