import numpy as np
from matplotlib import pyplot as plt

from src.logic.utils import generate_2d_graph
from scipy.signal import stft


def create_time_plot(x: np.ndarray, fs: float) -> plt.Figure:
    N = len(x)
    t = np.arange(N) / fs  # time vector in seconds

    # Use linear amplitude for time plot
    return generate_2d_graph(t, x, "Time [s]", "Amplitude", "Signal in Time Domain")


def create_frequency_plot(X_mag: np.ndarray, f: np.ndarray) -> plt.Figure:
    return generate_2d_graph(f, X_mag, "Frequency [Hz]", "Magnitude", "Signal in Frequency Domain")


def frequency_analysis(x, fs, only_positive=True, log_scale=False) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs frequency analysis on a given signal, computing its magnitude spectrum and
    corresponding frequency components. Allows customization for returning only positive
    frequencies and applying a logarithmic scale to the magnitude spectrum.

    :param x: 1D input signal for which the frequency analysis is performed.
    :param fs: Sampling frequency of the input signal.
    :param only_positive: Optional boolean flag; if True, returns only positive frequency
        components. Defaults to True.
    :param log_scale: Optional boolean flag; if True, applies a logarithmic scale to the
        magnitude spectrum. Defaults to False.
    :return: A tuple containing two numpy arrays:
        - The frequency components (`np.ndarray`).
        - The corresponding magnitude spectrum (`np.ndarray`).
    """
    N = len(x)
    X = np.fft.fft(x)
    X_mag = np.abs(X) / N

    f = np.fft.fftfreq(N, d=1/fs)

    if only_positive:
        idx = f >= 0
        f = f[idx]
        X_mag = X_mag[idx]

    if log_scale:
        X_mag = 20 * np.log10(X_mag + 1e-12)  # log(0) problem solved with +1e-12

    return f, X_mag

def plot_fft_3d(x: np.ndarray, fs: int) -> plt.Figure:
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
    # Get indices of top N magnitudes
    idx_topN = np.argsort(X_mag)[-N:]

    # Sort these indices by frequency
    idx_sorted_by_freq = idx_topN[np.argsort(f[idx_topN])]

    dominant_freqs = f[idx_sorted_by_freq]
    dominant_mags = X_mag[idx_sorted_by_freq]

    return dominant_freqs, dominant_mags


def compute_energy_power(x: np.ndarray) -> tuple[float, float]:
    energy = np.sum(x ** 2)  # sum of squared values
    N = len(x)
    power = energy / N

    return energy, power


def check_Parseval_theorem(x: np.ndarray, X: np.ndarray, tol: float = 1e-10) -> bool:
    N = len(x)
    energy_x = np.sum(np.abs(x) ** 2)
    energy_X = np.sum(np.abs(X) ** 2) / N

    print(f"Energy in time domain: {energy_x}")
    print(f"Energy in frequency domain: {energy_X}")
    return np.isclose(energy_x, energy_X, atol=tol)


def compute_short_time_fourier_transform(x: np.ndarray, fs: int, window_size: int = 256, overlap: int = 128):
    f, t, Zxx = stft(x, fs, nperseg=window_size, noverlap=overlap)
    return f, t, Zxx


def spectrogram_analysis(x: np.ndarray, fs: int, window_size: int = 256, overlap: int = 128,
                         title: str = "Spectrogram") -> plt.Figure:
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
    N = len(x)
    R = np.correlate(x, x, mode='full')  # full cross-correlation
    if normalize:
        R = R / R[N - 1]  # R[0] corresponds to zero lag in full mode
    return R


def autocorrelation_plot(x: np.ndarray, normalize: bool = True) -> plt.Figure:
    R = autocorrelation(x, normalize)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(R)
    ax.set_xlabel("Lag [samples]")
    ax.set_ylabel("Correlation")
    fig.tight_layout()
    return fig
