import numpy as np


def modulate_signal(x: np.ndarray, fs: int, fc: float, k: float = 1.0, mod_type: str = 'AM') -> np.ndarray:
    """
    Modulates an input signal using the specified modulation type. This function supports amplitude
    modulation (AM) and frequency modulation (FM).

    :param x: Input signal to be modulated.
    :type x: numpy.ndarray
    :param fs: Sampling frequency of the input signal, in Hz.
    :type fs: int
    :param fc: Carrier frequency for modulation, in Hz.
    :type fc: float
    :param k: Modulation index or sensitivity. Default is 1.0.
    :type k: float
    :param mod_type: Type of modulation to apply ('AM', or 'FM'). Case-insensitive.
    :type mod_type: str
    :return: Modulated signal.
    :rtype: numpy.ndarray
    """
    N = len(x)
    t = np.arange(N) / fs
    wc = 2 * np.pi * fc

    if mod_type.upper() == 'AM':  # Amplitude Modulation
        y = (1 + k * x) * np.cos(wc * t)
    elif mod_type.upper() == 'FM':  # Frequency modulation (cumulative sum for integral)
        y = np.cos(wc * t + k * np.cumsum(x) / fs)
    else:
        raise ValueError("mod_type must be 'AM' or 'FM'")

    return y
