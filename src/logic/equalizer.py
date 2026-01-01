import numpy as np

from src.logic.filters import fir_filter


def split_into_3bands(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the input signal into three frequency bands: low, mid, and high.

    This function uses FIR filters to divide the signal into three distinct
    frequency ranges. The low band covers 0-300 Hz, the mid band covers
    300-3000 Hz, and the high band starts from 3000 Hz upwards.

    :param x: Input signal as a NumPy array.
    :type x: np.ndarray
    :param fs: Sampling frequency of the input signal in Hz.
    :type fs: int
    :return: A tuple containing three arrays representing the low band,
        mid band, and high band components of the signal, respectively.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    # Low band: 0-300 Hz
    x_low = fir_filter(x, fs, cutoff=300, btype='low', numtaps=201)

    # Mid band: 300-3000 Hz
    x_mid = fir_filter(x, fs, cutoff=(300, 3000), btype='bandpass', numtaps=201)

    # High band: 3000-... Hz
    x_high = fir_filter(x, fs, cutoff=3000, btype='high', numtaps=201)

    return x_low, x_mid, x_high


def merge_bands(x_low: np.ndarray, x_mid: np.ndarray, x_high: np.ndarray) -> np.ndarray:
    """
    Merge three integer bands into a single value.

    This function takes three integer values representing separate numerical bands
    and merges them into a single combined value. The merging process is performed
    by summing the input values together. It is the user's responsibility to ensure
    that the input values represent meaningful data that should be merged in this
    manner.

    :param x_low: The value of the low band.
    :type x_low: np.ndarray
    :param x_mid: The value of the mid band.
    :type x_mid: np.ndarray
    :param x_high: The value of the high band.
    :type x_high: np.ndarray
    :return: The combined result of merging the three input bands.
    :rtype: np.ndarray
    """
    return x_low + x_mid + x_high


def equalize(x: np.ndarray, low_gain: float, mid_gain: float, high_gain: float, fs: int) -> np.ndarray:
    """
    Applies a 3-band equalization to the input signal by splitting it into low, mid,
    and high frequency bands, applying respective gain levels, and merging the bands
    back into a single signal.

    The function first decomposes the input signal into three distinct frequency bands,
    then scales each band by the provided gain factors, and finally combines the
    modified bands to produce the equalized output signal.

    :param x: Input audio signal represented as a numpy array.
    :param low_gain: Gain level applied to the low-frequency band.
    :param mid_gain: Gain level applied to the mid-frequency band.
    :param high_gain: Gain level applied to the high-frequency band.
    :param fs: Sampling frequency of the input audio signal.
    :return: Numpy array representing the equalized output signal.
    """
    x_low, x_mid, x_high = split_into_3bands(x, fs)
    return merge_bands(x_low * low_gain, x_mid * mid_gain, x_high * high_gain)
