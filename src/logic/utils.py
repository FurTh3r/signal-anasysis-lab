"""
utils.py

Module for generating and plotting simple graphs.
"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


def plot_graph(graph: plt.Figure):
    plt.figure(graph.number)
    plt.show()


def generate_2d_graph(x_values: np.ndarray, y_values: np.ndarray, x_label: str = "", y_label: str = "",
                      title: str = "") -> Figure:
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_two_graphs_side_by_side(x1, y1, title1, xlabel1, ylabel1,
                                 x2, y2, title2, xlabel2, ylabel2):
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


def plot_bands_frequency(f_l, X_l, f_m, X_m, f_h, X_h, fs):
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
