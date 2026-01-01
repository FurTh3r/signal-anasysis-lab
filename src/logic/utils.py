"""
utils.py

Module for generating and plotting simple graphs.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_graph(graph: plt.Figure):
    """
    Displays a given matplotlib figure in a window.

    This function takes a matplotlib figure object and renders it in a graphical
    window. It uses the provided figure's number attribute to focus the display
    on the relevant figure.

    :param graph: The matplotlib figure object to be displayed.
    :type graph: plt.Figure
    :return: None
    """
    plt.figure(graph.number)
    plt.show()


def generate_2d_graph(x_values: np.ndarray, y_values: np.ndarray, x_label: str = "", y_label: str = "",
                      title: str = "") -> Figure:
    """
    Generates a 2D graph using provided data points and labels for axes and title.

    This function creates a 2D line plot based on the given x and y values, labels
    the axes, sets a title for the graph, and applies a grid for better readability.
    The resulting figure object can be further manipulated or displayed using
    Matplotlib functionalities.

    :param x_values: The x-axis data points.
    :type x_values: numpy.ndarray
    :param y_values: The y-axis data points.
    :type y_values: numpy.ndarray
    :param x_label: The label for the x-axis. Defaults to an empty string.
    :type x_label: str
    :param y_label: The label for the y-axis. Defaults to an empty string.
    :type y_label: str
    :param title: The title of the graph. Defaults to an empty string.
    :type title: str
    :return: A Matplotlib figure object containing the generated 2D graph.
    :rtype: matplotlib.figure.Figure
    """
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_two_graphs_side_by_side(x1, y1, title1, xlabel1, ylabel1, x2, y2, title2, xlabel2, ylabel2):
    """
    Plots two graphs side by side within a single figure. Each graph is defined by its data points, title,
    x-axis label, and y-axis label.

    :param x1: Data points for the x-axis of the first graph.
    :param y1: Data points for the y-axis of the first graph.
    :param title1: Title of the first graph.
    :param xlabel1: Label for the x-axis of the first graph.
    :param ylabel1: Label for the y-axis of the first graph.
    :param x2: Data points for the x-axis of the second graph.
    :param y2: Data points for the y-axis of the second graph.
    :param title2: Title of the second graph.
    :param xlabel2: Label for the x-axis of the second graph.
    :param ylabel2: Label for the y-axis of the second graph.
    :return: None
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns

    # First plot
    axs[0].plot(x1, y1)
    axs[0].set_title(title1)
    axs[0].set_xlabel(xlabel1)
    axs[0].set_ylabel(ylabel1)
    axs[0].grid(True)

    # Second plot
    axs[1].plot(x2, y2)
    axs[1].set_title(title2)
    axs[1].set_xlabel(xlabel2)
    axs[1].set_ylabel(ylabel2)
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_bands_time(x_low, x_mid, x_high, fs):
    """
    Plots the time-domain representation of a signal split into frequency bands.

    This function takes three input signal components (low-frequency band, mid-frequency
    band, and high-frequency band) and plots them as individual time-domain signals. The
    function also includes labels, a legend, grid lines, and axis titles to provide an
    intuitive visualization of the signals over time.

    :param x_low: The low-frequency band component of the signal
    :param x_mid: The mid-frequency band component of the signal
    :param x_high: The high-frequency band component of the signal
    :param fs: Sampling frequency of the input signals
    :return: None
    """
    N = len(x_low)
    t = np.arange(N) / fs

    plt.figure(figsize=(12, 4))
    plt.plot(t, x_low, label="Low band (0–300 Hz)", alpha=0.8)
    plt.plot(t, x_mid, label="Mid band (300–3000 Hz)", alpha=0.8)
    plt.plot(t, x_high, label="High band (3000–8000 Hz)", alpha=0.8)

    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Signal split into frequency bands (time domain)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_bands_frequency(f_l, X_l, f_m, X_m, f_h, X_h):
    """
    Plots the frequency-domain representation of three audio frequency bands: low, mid, and high. Each band is characterized
    by its corresponding frequency values and magnitudes.

    :param f_l: Frequency values for the low frequency band (0–300 Hz)
    :type f_l: list or numpy.ndarray
    :param X_l: Magnitude values for the low frequency band
    :type X_l: list or numpy.ndarray
    :param f_m: Frequency values for the middle frequency band (300–3000 Hz)
    :type f_m: list or numpy.ndarray
    :param X_m: Magnitude values for the middle frequency band
    :type X_m: list or numpy.ndarray
    :param f_h: Frequency values for the high frequency band (3000–8000 Hz)
    :type f_h: list or numpy.ndarray
    :param X_h: Magnitude values for the high frequency band
    :type X_h: list or numpy.ndarray
    :return: None
    """
    plt.figure(figsize=(12, 4))
    plt.plot(f_l, X_l, label="Low band (0–300 Hz)")
    plt.plot(f_m, X_m, label="Mid band (300–3000 Hz)")
    plt.plot(f_h, X_h, label="High band (3000–8000 Hz)")

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.title("Frequency-domain representation of the three bands")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 9000)
    plt.tight_layout()
    plt.show()
