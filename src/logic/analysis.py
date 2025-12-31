import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import stft

from src.logic.utils import generate_2d_graph


def create_time_plot(x: np.ndarray, fs: float) -> plt.Figure:
    """
    Creates a time-domain plot of a given signal array.

    This function generates a 2D graph representing the signal in the time domain,
    using the input signal array and its sampling frequency. The x-axis represents time
    (in seconds), and the y-axis represents the amplitude of the signal. The plot is returned
    as a `matplotlib.figure.Figure` object.

    :param x: Signal array, representing the amplitude of the signal over time.
    :type x: np.ndarray
    :param fs: Sampling frequency of the signal, in hertz (Hz).
    :type fs: float
    :return: A matplotlib figure containing the generated time-domain signal plot.
    :rtype: plt.Figure
    """
    N = len(x)
    t = np.arange(N) / fs  # time vector in seconds

    # Use linear amplitude for time plot
    return generate_2d_graph(t, x, "Time [s]", "Amplitude", "Signal in Time Domain")


def create_frequency_plot(X_mag: np.ndarray, f: np.ndarray) -> plt.Figure:
    """
    Creates and returns a frequency plot representing the relationship
    between frequency and magnitude in the frequency domain.

    :param X_mag: Magnitude values of the signal in the frequency domain.
    :type X_mag: numpy.ndarray
    :param f: Corresponding frequency values for the signal in the frequency domain.
    :type f: numpy.ndarray
    :return: A matplotlib Figure object containing the frequency plot.
    :rtype: matplotlib.figure.Figure
    """
    return generate_2d_graph(f, X_mag, "Frequency [Hz]", "Magnitude", "Signal in Frequency Domain")


def frequency_analysis(x, fs, only_positive=True) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs frequency analysis on a given signal, computing its FFT and
    corresponding frequency components. Allows customization for returning only positive
    frequencies. Log scale is ignored if returning complex FFT.

    :param x: 1D input signal for which the frequency analysis is performed.
    :param fs: Sampling frequency of the input signal.
    :param only_positive: Optional boolean flag; if True, returns only positive frequency
        components. Defaults to True.
    :return: A tuple containing two numpy arrays:
        - The frequency components (`np.ndarray`).
        - The FFT values (`np.ndarray`, complex).
    """
    N = len(x)
    X = np.fft.fft(x)
    f = np.fft.fftfreq(N, d=1 / fs)

    if only_positive:
        idx = f >= 0
        f = f[idx]
        X = X[idx]

    return f, X


def plot_fft_3d(x: np.ndarray, fs: int) -> plt.Figure:
    """
    This function generates a 3D plot of the Discrete Fourier Transform (DFT) of a given signal. It represents
    the frequency components as a 3D helix where the x-axis corresponds to frequency, the y-axis to the real
    part of the DFT, and the z-axis to the imaginary part of the DFT. This visualization is useful for analyzing
    the frequency domain of a one-dimensional signal in a compact and intuitive manner.

    :param x: A 1D NumPy array representing the discrete signal to be transformed.
    :param fs: Sampling frequency of the signal in hertz.
    :return: A Matplotlib figure object containing the 3D plot of the DFT.
    """
    N = len(x)
    X = np.fft.fft(x)
    f = np.fft.fftfreq(N, d=1 / fs)

    # Frequenze positive
    idx = np.where(f >= 0)
    f = f[idx]
    X = X[idx]

    Re_X = np.real(X)
    Im_X = np.imag(X)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(f, Re_X, Im_X, color='blue', lw=2)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Re(X)")
    ax.set_zlabel("Im(X)")
    ax.set_title("3D Helix of DFT")
    fig.tight_layout()

    return fig


def get_dominant_frequencies(f: np.ndarray, X_mag: np.ndarray, N: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Identifies the dominant frequencies and their respective magnitudes from a set of
    frequencies and magnitudes captured in frequency domain data. The function returns
    the top `N` dominant frequencies sorted in increasing order, along with their
    corresponding magnitudes.

    :param f: A numpy array containing the frequency components.
    :param X_mag: A numpy array containing the corresponding magnitudes of the frequency components.
    :param N: An integer specifying the number of dominant frequencies to retrieve. Default is 10.
    :return: A tuple containing two numpy arrays:
        - The first array contains the `N` dominant frequencies in ascending order.
        - The second array contains the magnitudes corresponding to the `N` dominant frequencies.
    """
    # Get indices of top N magnitudes
    idx_topN = np.argsort(X_mag)[-N:]

    # Sort these indices by frequency
    idx_sorted_by_freq = idx_topN[np.argsort(f[idx_topN])]

    dominant_freqs = f[idx_sorted_by_freq]
    dominant_mags = X_mag[idx_sorted_by_freq]

    return dominant_freqs, dominant_mags


def compute_energy_power(x: np.ndarray) -> tuple[float, float]:
    """
    Compute the total energy and average power of a signal.

    This function calculates the energy and power of a given signal
    represented as a numpy array. The energy is computed as the sum of
    the squares of the signal values, and the power is obtained as the
    average of the energy normalized by the number of samples in the
    signal.

    :param x: The input signal represented as a numpy array.
    :type x: numpy.ndarray
    :return: A tuple containing two elements. The first element is the total
        energy (float), and the second element is the average power (float)
        of the signal.
    :rtype: tuple[float, float]
    """
    energy = np.sum(x ** 2)  # sum of squared values
    N = len(x)
    power = energy / N

    return energy, power


def check_parseval_theorem(x: np.ndarray, X: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Checks Parseval's theorem for a given time-domain signal and its frequency-domain
    representation. Parseval's theorem states that the total energy of the signal in
    the time domain is equal to the total energy of its spectrum in the frequency domain.

    :param x: The signal in the time domain represented as a 1-dimensional NumPy array.
    :param X: The spectrum (Fourier transform) of the signal represented as a 1-dimensional
               NumPy array.
    :param tol: The tolerance value for comparing the energies. Defaults to 1e-10.
    :return: A boolean indicating whether the energies are approximately equal within
             the specified tolerance.
    """
    N = len(x)
    energy_x = np.sum(np.abs(x) ** 2)
    energy_X = np.sum(np.abs(X) ** 2) / N

    print(f"Energy in time domain: {energy_x}")
    print(f"Energy in frequency domain: {energy_X}")
    return np.isclose(energy_x, energy_X, atol=tol)


def compute_short_time_fourier_transform(x: np.ndarray, fs: int, window_size: int = 256, overlap: int = 128):
    """
    Compute the Short-Time Fourier Transform (STFT) of a signal.

    This function computes the STFT of a one-dimensional time-domain signal using
    the specified sampling frequency, window size, and overlap. The STFT provides
    a time-frequency representation of the signal, breaking it into small time
    segments and computing the Fourier Transform for each segment.

    :param x: Signal to be transformed as a 1-dimensional numpy ndarray.
    :param fs: Sampling frequency of the signal in Hertz.
    :param window_size: Size of the window applied to each segment (default: 256).
    :param overlap: Number of overlapping points between consecutive windows
        (default: 128).
    :return: A tuple containing:
        - f: Array of sample frequencies (in Hertz).
        - t: Array of segment times (in seconds).
        - Zxx: STFT of x with time and frequency as its dimensions.
    """
    f, t, Zxx = stft(x, fs, nperseg=window_size, noverlap=overlap)
    return f, t, Zxx


def spectrogram_analysis(x: np.ndarray, fs: int, window_size: int = 256, overlap: int = 128,
                         title: str = "Spectrogram") -> plt.Figure:
    """
    Analyzes the spectrogram of a given signal using the Short-Time Fourier Transform (STFT) and
    generates a visual representation of its frequency content over time. The spectrogram is displayed
    in decibels (dB) scale, with a color bar indicating the magnitude values.

    :param x: The input signal to analyze.
    :type x: numpy.ndarray
    :param fs: The sampling frequency of the input signal in Hz.
    :type fs: int
    :param window_size: The size of the window for the STFT in samples. Defaults to 256.
    :type window_size: int, optional
    :param overlap: The overlap between consecutive windows in samples. Defaults to 128.
    :type overlap: int, optional
    :param title: The title for the spectrogram plot. Defaults to "Spectrogram".
    :type title: str, optional
    :return: A matplotlib figure containing the spectrogram plot.
    :rtype: matplotlib.figure.Figure
    """
    f, t, Zxx = compute_short_time_fourier_transform(x, fs, window_size, overlap)
    magnitude = np.abs(Zxx)

    fig, ax = plt.subplots(figsize=(10, 4))
    c = ax.pcolormesh(t, f, 20 * np.log10(magnitude + 1e-12), shading='gouraud')  # dB scale
    ax.set_title(title)
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel("Time [s]")
    fig.colorbar(c, ax=ax, label="Magnitude [dB]")
    ax.set_ylim(0, fs / 2)
    fig.tight_layout()

    return fig


def autocorrelation(x: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Computes the autocorrelation of a given 1D numpy array. Autocorrelation measures
    the correlation of a signal with a delayed version of itself over varying time lags.
    Optionally normalizes the result by the zero-lag value.

    :param x: The input array for which autocorrelation is to be computed.
    :param normalize: A boolean flag. If True, the output is normalized by the
        zero-lag value of the autocorrelation, which corresponds to the maximal
        value of the signal's energy.
    :return: An array containing the autocorrelation values for the given input
        signal, computed for all lags in "full" mode.
    """
    N = len(x)
    R = np.correlate(x, x, mode='full')  # full cross-correlation
    if normalize:
        R = R / R[N - 1]  # R[0] corresponds to zero lag in full mode
    return R


def autocorrelation_plot(x: np.ndarray, normalize: bool = True) -> plt.Figure:
    """
    Generates an autocorrelation plot for a given time series using its computed
    autocorrelation values. The plot illustrates the correlation of the series with
    itself at different time lags, providing an insightful visual representation
    of repetitive patterns or dependencies in the data.

    :param x: The input time series data as a numpy array.
    :param normalize: A boolean flag to indicate whether the autocorrelation values
        should be normalized to the range [-1, 1]. Defaults to True.
    :return: A matplotlib Figure object containing the autocorrelation plot.
    """
    R = autocorrelation(x, normalize)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(R)
    ax.set_xlabel("Lag [samples]")
    ax.set_ylabel("Correlation")
    fig.tight_layout()
    return fig
