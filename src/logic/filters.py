import numpy as np
from scipy.signal import firwin, lfilter


def fir_filter(x: np.ndarray, fs: int, cutoff: int, btype: str = 'low', numtaps: int = 101):
    """
    Designs and applies a Finite Impulse Response (FIR) filter to the input signal. The function allows for the creation
    of low-pass, high-pass, or band-pass filters based on the specified parameters. It leverages the window method
    to compute the filter coefficients and subsequently applies the filter to the input signal.

    :param x: Input signal to be filtered.
    :type x: numpy.ndarray
    :param fs: Sampling frequency of the input signal.
    :type fs: int
    :param cutoff: The cutoff frequency or frequencies, depending on the type of filter specified. For band-pass
        filters, it should be a sequence of frequencies [low_cutoff, high_cutoff].
    :type cutoff: int or list[int]
    :param btype: Type of filter to apply. Must be one of the following: 'low' for low-pass, 'high' for high-pass,
        or 'bandpass' for band-pass.
    :type btype: str
    :param numtaps: Number of filter coefficients (taps) to design. Determines the order of the FIR filter.
    :type numtaps: int
    :return: Filtered output signal.
    :rtype: numpy.ndarray
    """
    nyq = 0.5 * fs  # Nysquist frequency

    # Design FIR coefficients
    if btype == 'low':
        b = firwin(numtaps, cutoff / nyq)
    elif btype == 'high':
        b = firwin(numtaps, cutoff / nyq, pass_zero=False)
    elif btype == 'bandpass':
        b = firwin(numtaps, [c / nyq for c in cutoff], pass_zero=False)
    else:
        raise ValueError("btype must be 'low', 'high', or 'bandpass'")

    # Apply filter
    y = lfilter(b, 1.0, x)
    return y
