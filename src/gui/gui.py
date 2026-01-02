import threading
import tkinter as tk
from tkinter import ttk, filedialog

import matplotlib.pyplot as plt
import messagebox
import numpy as np
import sounddevice as sd
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

from src.logic.IO import load_audio, save_audio
from src.logic.analysis import frequency_analysis, compute_energy_power, get_dominant_frequencies, autocorrelation, \
    compute_short_time_fourier_transform
from src.logic.equalizer import equalize, split_into_3bands
from src.logic.modulation import modulate_signal

# =========================
# Color Palettes
# =========================
BG_MAIN = "#eaf4fb"
BG_FRAME = "#dbeeff"
ACCENT = "#4aa3df"
ACCENT_DARK = "#2b7bb9"
TEXT_COLOR = "#1f2d3d"


def _create_vertical_slider(parent, label, default):
    """
    Creates a vertical slider UI control with a label and a value display.

    This function initializes and returns a vertical slider widget enclosed in a
    frame. The label displays the given text, and the value label dynamically
    updates to reflect the current value of the slider. The slider can range
    from 2.0 to 0.0 with a specified resolution of 0.01.

    :param parent: The parent widget where the slider frame will be placed.
    :type parent: tkinter widget
    :param label: The text to display as the label for the slider.
    :type label: str
    :param default: The initial value for the slider.
    :type default: float
    :return: The initialized vertical slider widget.
    :rtype: tkinter.Scale
    """
    frame = ttk.Frame(parent)
    frame.pack(side="left", padx=15)

    ttk.Label(frame, text=label).pack()
    value_label = ttk.Label(frame, text=f"{default:.2f}")
    value_label.pack()

    slider = tk.Scale(frame, from_=2.0, to=0.0, resolution=0.01, orient="vertical", length=150,
                      command=lambda v: value_label.config(text=f"{float(v):.2f}"))
    slider.set(default)
    slider.pack()
    return slider


def plot_graph(x_values, y_values, x_name, y_name, ax, canvas, title="Signal Graph"):
    if x_values is None or y_values is None or ax is None or canvas is None:
        return

    ax.clear()
    ax.set_xlim(0, x_values[-1])
    ax.set_ylim(1.1 * np.min(y_values), 1.1 * np.max(y_values))
    ax.plot(x_values, y_values)
    ax.set_title(title)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.grid(True)

    canvas.draw_idle()


def plot_3graph(x1, y1, x2, y2, x3, y3, ax, canvas, title="Signal Graph", xlabel="X-axis", ylabel="Y-axis",
                labels=("Series 1", "Series 2", "Series 3")):
    """
    Plots three graphs on a single axis, adjusting axis limits, labels, and titles. It also refreshes a provided canvas for
    display updates.

    :param x1: First set of x-axis values
    :type x1: array-like or list
    :param y1: First set of y-axis values
    :type y1: array-like or list
    :param x2: Second set of x-axis values
    :type x2: array-like or list
    :param y2: Second set of y-axis values
    :type y2: array-like or list
    :param x3: Third set of x-axis values
    :type x3: array-like or list
    :param y3: Third set of y-axis values
    :type y3: array-like or list
    :param ax: The matplotlib axis object to plot the graphs on
    :type ax: matplotlib.axes.Axes
    :param canvas: The canvas associated with the axis for refreshing the display
    :type canvas: matplotlib.backend_bases.FigureCanvasBase
    :param title: Title of the graph, defaults to "Signal Graph"
    :type title: str, optional
    :param xlabel: Label for the x-axis, defaults to "X-axis"
    :type xlabel: str, optional
    :param ylabel: Label for the y-axis, defaults to "Y-axis"
    :type ylabel: str, optional
    :param labels: Tuple containing labels for the three series, defaults to ("Series 1", "Series 2", "Series 3")
    :type labels: tuple, optional
    :return: None
    """
    if (
            x1 is None or y1 is None or x2 is None or y2 is None or x3 is None or y3 is None or ax is None or canvas is None):
        return

    ax.clear()

    # Determinating the limits of the x-axis
    xmin = min(np.min(x1), np.min(x2), np.min(x3))
    xmax = max(np.max(x1), np.max(x2), np.max(x3))
    ax.set_xlim(xmin, xmax)

    # Determinating the limits of the y-axis
    ymin = min(np.min(y1), np.min(y2), np.min(y3))
    ymax = max(np.max(y1), np.max(y2), np.max(y3))
    ax.set_ylim(1.1 * ymin, 1.1 * ymax)

    # Plot of the three signals
    ax.plot(x1, y1, label=labels[0])
    ax.plot(x2, y2, label=labels[1])
    ax.plot(x3, y3, label=labels[2])

    # Titles and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()

    canvas.draw_idle()


def generate_sinusoid(frequency, amplitude, phase, duration, fs):
    """
    Generates a sinusoidal signal.

    :param frequency: Frequency of the sinusoid in Hz
    :param amplitude: Amplitude of the sinusoid
    :param phase: Phase of the sinusoid in radians
    :param duration: Duration of the signal in seconds
    :param fs: Sampling frequency in Hz
    :return: numpy array containing the sinusoidal signal
    """
    t = np.arange(0, duration, 1 / fs)
    signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return signal, t


def show_error(message: str, title: str = "Error"):
    """
    Displays an error message in a message box with the provided title.

    This function utilizes the `messagebox.showerror` method to display
    an error dialog window with the specified message and an optional
    title. The title defaults to "Error" if not provided.

    :param message: The error message to be displayed in the dialog.
    :type message: str
    :param title: The title of the message box. Defaults to "Error".
    :type title: str
    :return: None
    """
    messagebox.showerror(title, message)


class AudioDSPApp(tk.Tk):
    """
    The main application class for a graphical user interface (GUI) toolkit designed for audio DSP (Digital Signal Processing).
    This class inherits from `tk.Tk` and serves as the entry point for creating and managing the GUI components, including
    audio control and signal analysis features.
    """

    def __init__(self):
        """
        Initializes the main application for the Audio DSP Toolkit.
        """
        super().__init__()

        self.title("Audio DSP Toolkit")
        self.configure(bg=BG_MAIN)

        # Component Variables:
        self.n_max = 10  # Number of top frequencies to show

        self.volume_value_label = None
        self.loop_button = None
        self.volume_slider = None
        self.fig_freq = None
        self.fig_3d = None
        self.fig_spec = None
        self.canvas_time = None
        self.canvas_freq = None
        self.canvas_3d = None
        self.canvas_spec = None
        self.fig_time = None
        self.k_slider = None
        self.fc_slider = None
        self.mod_type_var = None
        self.high_gain = None
        self.mid_gain = None
        self.low_gain = None
        self.equalizer_enabled = tk.BooleanVar(value=True)
        self.modulation_enabled = tk.BooleanVar(value=True)
        self.ax_time = None
        self.ax_freq = None
        self.frame_graphs = None
        self.canvas_freq2 = None
        self.ax_freq2 = None
        self.fig_freq2 = None
        self.canvas_time2 = None
        self.ax_time2 = None
        self.fig_time2 = None
        self.dur_entry = None
        self.phase_entry = None
        self.amp_entry = None
        self.freq_entry = None
        self.graph_choice = None
        self.logMagnitude = tk.BooleanVar(value=True)
        self.freq_text = None
        self.parseval_var = None
        self.power_var = None
        self.energy_var = None
        self.canvas_autocorr = None
        self.fig_autocorr = None
        self.ax_spectro = None
        self.canvas_spectro = None
        self.ax_autocorr = None
        self.fig_spectro = None
        self.top_freqs_mod_var = None
        self.power_mod_var = None
        self.power_orig_var = None
        self.energy_mod_var = None
        self.energy_orig_var = None
        self.top_freqs_orig_var = None
        self.freq_text_orig = None
        self.freq_text_mod = None
        self._spectro_colorbar = None

        # =========================
        # Style Settings
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("TFrame", background=BG_MAIN)
        style.configure("TLabelframe", background=BG_FRAME)
        style.configure("TLabelframe.Label", background=BG_FRAME, foreground=TEXT_COLOR, font=("Segoe UI", 10, "bold"))

        style.configure("TButton", background=ACCENT, foreground="white", padding=6, font=("Segoe UI", 9))
        style.map("TButton", background=[("active", ACCENT_DARK)])

        style.configure("TNotebook", background=BG_MAIN, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG_FRAME, foreground=TEXT_COLOR, padding=(12, 6))
        style.map("TNotebook.Tab", background=[("selected", ACCENT)], foreground=[("selected", "white")])

        # =========================
        # Audio Variables
        self.audio_signal = None
        self.audio_signal_modified = None
        self.fs = 44100
        self.audio_thread = None
        self.stop_event = threading.Event()
        self.loop_enabled = False

        # =========================
        # Notebook
        self.tab_control = ttk.Notebook(self)
        self.tab_control.pack(expand=1, fill="both")

        self.tab_control_tab1 = ttk.Frame(self.tab_control)
        self.tab_control_tab2 = ttk.Frame(self.tab_control)

        self.tab_control.add(self.tab_control_tab1, text='Audio Elaboration')
        self.tab_control.add(self.tab_control_tab2, text='Signal Analysis')

        self.create_signal_analysis_tab()
        self.create_audio_control_tab()

    # =========================
    # TAB 1 – Audio Control
    # =========================
    def create_audio_control_tab(self):
        # =======================
        # Main Layout
        # =======================
        main_frame = ttk.Frame(self.tab_control_tab1)
        main_frame.pack(expand=True, fill="both", padx=10, pady=10)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side="left", fill="y", padx=10)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side="left", expand=True, fill="both", padx=10)

        graphs_frame = ttk.Frame(right_frame)
        graphs_frame.pack(expand=True, fill="both")

        graphs_frame.columnconfigure(0, weight=1)
        graphs_frame.rowconfigure(0, weight=1)
        graphs_frame.rowconfigure(2, weight=1)

        graphs_frame.columnconfigure(1, weight=1)
        graphs_frame.rowconfigure(0, weight=1)
        graphs_frame.rowconfigure(2, weight=1)

        # =======================
        # Audio Controls
        # =======================
        frame_io = ttk.Labelframe(left_frame, text="Audio I/O")
        frame_io.pack(fill="x", pady=10)

        # Load / Save
        top_frame = ttk.Frame(frame_io)
        top_frame.pack(fill="x", pady=5)
        ttk.Button(top_frame, text="Load", command=self.load_audio).pack(side="left", padx=5)
        ttk.Button(top_frame, text="Save", command=self.save_audio_file).pack(side="left", padx=5)

        # Record / Play / Stop / Loop
        bottom_frame = ttk.Frame(frame_io)
        bottom_frame.pack(fill="x", pady=5)
        ttk.Button(bottom_frame, text="Play", command=self.play_audio).pack(side="left", padx=5)
        ttk.Button(bottom_frame, text="Stop", command=self.stop_audio).pack(side="left", padx=5)

        self.loop_button = ttk.Button(bottom_frame, text="Loop OFF", command=self.loop_play_audio)
        self.loop_button.pack(side="left", padx=5)

        # Graph selection buttons
        button_frame = ttk.LabelFrame(self, text="Graph Selection")
        button_frame.pack(padx=10, pady=10, fill="x")

        self.graph_choice = tk.StringVar(value="magnitude")

        # Radio buttons in orizzontale
        ttk.Radiobutton(button_frame, text="FFT Phase", variable=self.graph_choice, value="phase",
                        command=self.update_plot).pack(side="left", padx=5, pady=5)
        ttk.Radiobutton(button_frame, text="FFT Module", variable=self.graph_choice, value="magnitude",
                        command=self.update_plot).pack(side="left", padx=5, pady=5)
        ttk.Radiobutton(button_frame, text="3 Bande (Low/Medium/High)", variable=self.graph_choice, value="3bands",
                        command=self.update_plot).pack(side="left", padx=5, pady=5)

        ttk.Checkbutton(button_frame, text="Enable Log Scale", variable=self.logMagnitude).pack(side="left", padx=10)

        ttk.Button(button_frame, text="Reload Graphs", command=self.update_plot).pack(side="left", padx=5)
        # =======================
        # Generate Sinusoid
        # =======================
        frame_generate = ttk.Labelframe(left_frame, text="Generate Signal")
        frame_generate.pack(fill="x", pady=10)

        # Frequency
        freq_frame = ttk.Frame(frame_generate)
        freq_frame.pack(fill="x", pady=2)
        ttk.Label(freq_frame, text="Frequency [Hz]").pack(side="left")
        self.freq_entry = ttk.Entry(freq_frame, width=10)
        self.freq_entry.insert(0, "1000")  # valore di default
        self.freq_entry.pack(side="left", padx=5)

        # Amplitude
        amp_frame = ttk.Frame(frame_generate)
        amp_frame.pack(fill="x", pady=2)
        ttk.Label(amp_frame, text="Amplitude").pack(side="left")
        self.amp_entry = ttk.Entry(amp_frame, width=10)
        self.amp_entry.insert(0, "1.0")  # valore di default
        self.amp_entry.pack(side="left", padx=5)

        # Phase
        phase_frame = ttk.Frame(frame_generate)
        phase_frame.pack(fill="x", pady=2)
        ttk.Label(phase_frame, text="Phase [rad]").pack(side="left")
        self.phase_entry = ttk.Entry(phase_frame, width=10)
        self.phase_entry.insert(0, "0.0")  # valore di default
        self.phase_entry.pack(side="left", padx=5)

        # Duration
        dur_frame = ttk.Frame(frame_generate)
        dur_frame.pack(fill="x", pady=2)
        ttk.Label(dur_frame, text="Duration [s]").pack(side="left")
        self.dur_entry = ttk.Entry(dur_frame, width=10)
        self.dur_entry.insert(0, "1.0")
        self.dur_entry.pack(side="left", padx=5)

        # Pulsante Generate
        ttk.Button(frame_generate, text="Generate", command=self.generate_sinusoid).pack(pady=5)

        # Volume
        vol_frame = ttk.Frame(frame_io)
        vol_frame.pack(side="left", padx=15)
        ttk.Label(vol_frame, text="Volume").pack()
        self.volume_value_label = ttk.Label(vol_frame, text="1.00")
        self.volume_value_label.pack()

        self.volume_slider = tk.Scale(vol_frame, from_=1.0, to=0.0, resolution=0.01, orient="vertical", length=120,
                                      command=self.on_volume_change  # callback
                                      )
        self.volume_slider.set(1.0)
        self.volume_slider.pack()

        # =======================
        # Equalizer
        frame_eq = ttk.Labelframe(left_frame, text="Equalizer")
        frame_eq.pack(fill="x", pady=10)

        self.low_gain = _create_vertical_slider(frame_eq, "Low", 1.0)
        self.mid_gain = _create_vertical_slider(frame_eq, "Mid", 1.0)
        self.high_gain = _create_vertical_slider(frame_eq, "High", 1.0)
        ttk.Checkbutton(frame_eq, text="Enable EQ", variable=self.equalizer_enabled).pack(side="left", padx=10)

        # =======================
        # Modulation
        frame_mod = ttk.Labelframe(left_frame, text="Modulation")
        frame_mod.pack(fill="x", pady=10)

        self.mod_type_var = tk.StringVar(value="AM")
        ttk.Label(frame_mod, text="Tipo").pack(side="left")
        ttk.OptionMenu(frame_mod, self.mod_type_var, "AM", "AM", "FM").pack(side="left", padx=5)

        ttk.Label(frame_mod, text="f_c [Hz]").pack(side="left", padx=(10, 2))
        self.fc_slider = tk.Scale(frame_mod, from_=100, to=20000, orient="horizontal", length=180, resolution=1)
        self.fc_slider.set(5000)
        self.fc_slider.pack(side="left", fill="x", expand=True)

        ttk.Label(frame_mod, text="k").pack(side="left", padx=(10, 2))
        self.k_slider = tk.Scale(frame_mod, from_=0.0, to=5.0, orient="horizontal", length=120, resolution=0.01)
        self.k_slider.set(0.5)
        self.k_slider.pack(side="left")

        ttk.Checkbutton(frame_mod, text="Enable Modulation", variable=self.modulation_enabled).pack(side="left",
                                                                                                    padx=10)
        # =======================
        # Apply Processing
        ttk.Button(left_frame, text="Apply Processing", command=self.apply_processing).pack(pady=10)

        # ===== TIME GRAPH ORIGINAL =====
        self.fig_time = plt.Figure(figsize=(4, 3))
        self.ax_time = self.fig_time.add_subplot(111)
        self.canvas_time = FigureCanvasTkAgg(self.fig_time, master=graphs_frame)
        self.canvas_time.get_tk_widget().grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        toolbar_time = NavigationToolbar2Tk(self.canvas_time, graphs_frame, pack_toolbar=False)
        toolbar_time.update()
        toolbar_time.grid(row=1, column=0, sticky="w")

        # ===== FREQUENCY GRAPH ORIGINAL =====
        self.fig_freq = plt.Figure(figsize=(4, 3))
        self.ax_freq = self.fig_freq.add_subplot(111)
        self.canvas_freq = FigureCanvasTkAgg(self.fig_freq, master=graphs_frame)
        self.canvas_freq.get_tk_widget().grid(row=2, column=0, padx=5, pady=5, sticky="nsew")

        toolbar_freq = NavigationToolbar2Tk(self.canvas_freq, graphs_frame, pack_toolbar=False)
        toolbar_freq.update()
        toolbar_freq.grid(row=3, column=0, sticky="w")

        # ===== TIME GRAPH EDITED =====
        self.fig_time2 = plt.Figure(figsize=(4, 3))
        self.ax_time2 = self.fig_time2.add_subplot(111)
        self.canvas_time2 = FigureCanvasTkAgg(self.fig_time2, master=graphs_frame)
        self.canvas_time2.get_tk_widget().grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        toolbar_time2 = NavigationToolbar2Tk(self.canvas_time2, graphs_frame, pack_toolbar=False)
        toolbar_time2.update()
        toolbar_time2.grid(row=1, column=1, sticky="w")

        # ===== FREQUENCY GRAPH EDITED =====
        self.fig_freq2 = plt.Figure(figsize=(4, 3))
        self.ax_freq2 = self.fig_freq2.add_subplot(111)
        self.canvas_freq2 = FigureCanvasTkAgg(self.fig_freq2, master=graphs_frame)
        self.canvas_freq2.get_tk_widget().grid(row=2, column=1, padx=5, pady=5, sticky="nsew")

        toolbar_freq2 = NavigationToolbar2Tk(self.canvas_freq2, graphs_frame, pack_toolbar=False)
        toolbar_freq2.update()
        toolbar_freq2.grid(row=3, column=1, sticky="w")

    # =========================
    # TAB 2 – Signal Analysis
    # =========================
    def create_signal_analysis_tab(self):
        ttk.Label(self.tab_control_tab2, text="Advanced analisys Tab", font=("Segoe UI", 12, "bold")).pack(pady=10)

        # =========================
        # Graph Frame
        # =========================
        graphs_frame = ttk.Frame(self.tab_control_tab2)
        graphs_frame.pack(fill="both", expand=True, padx=10, pady=5)

        graphs_frame.columnconfigure(0, weight=1)
        graphs_frame.columnconfigure(1, weight=1)

        # --- STFT spectrum
        spectro_frame = ttk.LabelFrame(graphs_frame, text="Spectrogram (STFT)")
        spectro_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.fig_spectro, self.ax_spectro = plt.subplots(figsize=(5, 3))
        self.canvas_spectro = FigureCanvasTkAgg(self.fig_spectro, master=spectro_frame)
        self.canvas_spectro.get_tk_widget().pack(fill="both", expand=True)

        # --- Autocorrelation
        autocorr_frame = ttk.LabelFrame(graphs_frame, text="Autocorrelation")
        autocorr_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.fig_autocorr, self.ax_autocorr = plt.subplots(figsize=(5, 3))
        self.canvas_autocorr = FigureCanvasTkAgg(self.fig_autocorr, master=autocorr_frame)
        self.canvas_autocorr.get_tk_widget().pack(fill="both", expand=True)

        # =========================
        # Frame numeric analysis
        analysis_frame = ttk.LabelFrame(self.tab_control_tab2, text="Numerical Analysis of the Signal")
        analysis_frame.pack(fill="x", padx=10, pady=10)

        analysis_frame.columnconfigure(1, weight=1)

        # Energy & Power
        def create_label_pair(frame, row, text, var):
            ttk.Label(frame, text=text).grid(row=row, column=0, sticky="w", padx=5)
            ttk.Label(frame, textvariable=var).grid(row=row, column=1, sticky="w")

        self.energy_orig_var = tk.StringVar(value="—")
        create_label_pair(analysis_frame, 0, "Original Signal Energy:", self.energy_orig_var)

        self.energy_mod_var = tk.StringVar(value="—")
        create_label_pair(analysis_frame, 1, "Modified Signal Energy:", self.energy_mod_var)

        self.power_orig_var = tk.StringVar(value="—")
        create_label_pair(analysis_frame, 2, "Original Signal Power:", self.power_orig_var)

        self.power_mod_var = tk.StringVar(value="—")
        create_label_pair(analysis_frame, 3, "Modified Signal Power:", self.power_mod_var)

        # =========================
        # Dominant frequencies
        freq_frame = ttk.LabelFrame(self.tab_control_tab2, text="Dominant Frequencies")
        freq_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Original signal
        ttk.Label(freq_frame, text="Original Signal:").pack(anchor="w", padx=5)
        self.freq_text_orig = tk.Text(freq_frame, height=6, wrap="word")
        self.freq_text_orig.pack(fill="both", expand=True, padx=5, pady=2)
        self.freq_text_orig.configure(state="disabled")

        # Edited signal
        ttk.Label(freq_frame, text="Edited Signal:").pack(anchor="w", padx=5)
        self.freq_text_mod = tk.Text(freq_frame, height=6, wrap="word")
        self.freq_text_mod.pack(fill="both", expand=True, padx=5, pady=2)
        self.freq_text_mod.configure(state="disabled")

    # =========================
    # Audio GUI logic
    # =========================
    def calculate_spectrogram(self):
        """
        Calculates and updates the spectrogram of the modified audio signal
        using an object-oriented Matplotlib approach (GUI-safe).
        """
        if self.audio_signal_modified is None:
            return

        window_size = 256
        overlap = 128

        x = self.audio_signal_modified
        fs = self.fs

        f, t, Zxx = compute_short_time_fourier_transform(x, fs, window_size=window_size, overlap=overlap)

        magnitude_db = 20 * np.log10(np.abs(Zxx) + 1e-12)

        self.ax_spectro.clear()
        pcm = self.ax_spectro.pcolormesh(t, f, magnitude_db, shading="gouraud")

        self.ax_spectro.set_title("Spectrogram (Modified Signal)")
        self.ax_spectro.set_xlabel("Time [s]")
        self.ax_spectro.set_ylabel("Frequency [Hz]")
        self.ax_spectro.set_ylim(0, fs / 2)

        if not hasattr(self, "_spectro_colorbar") or self._spectro_colorbar is None:
            self._spectro_colorbar = self.fig_spectro.colorbar(pcm, ax=self.ax_spectro, label="Magnitude [dB]")
        else:
            self._spectro_colorbar.update_normal(pcm)

        self.fig_spectro.tight_layout()
        self.canvas_spectro.draw_idle()

    def calculate_autocorrelation(self):
        """
        Calculates and visualizes the autocorrelation of a modified audio signal.

        This method computes the autocorrelation of the modified audio signal stored
        in the instance and plots the resulting data. The autocorrelation indicates
        how correlated the signal is with a time-shifted version of itself at
        different lag periods. It is normalized to assess similarity independent of
        signal amplitude, and the visualization is rendered on the associated plot.

        :raises RuntimeError: If the modified audio signal is not available or has not
            been initialized.
        """
        if self.audio_signal_modified is None:
            return

        R = autocorrelation(self.audio_signal_modified)
        t = np.arange(-len(self.audio_signal_modified) + 1, len(self.audio_signal_modified)) / self.fs

        self.ax_autocorr.clear()
        self.ax_autocorr.plot(t, R)
        self.ax_autocorr.set_title("Autocorrelation (Modified Signal)")
        self.ax_autocorr.set_xlabel("Lag [s]")
        self.ax_autocorr.set_ylabel("Normalized Amplitude")
        self.ax_autocorr.grid(True)

        self.fig_autocorr.tight_layout()
        self.canvas_autocorr.draw_idle()

    def calculate_dominant_freqs(self):
        """
        Analyzes the frequency components of the original and modified audio signals and updates the
        associated GUI text fields with the dominant frequencies.

        This method processes both the original and edited audio signals to determine their dominant
        frequencies using frequency analysis and updates the designated text fields in the application
        to display the results. If no audio signal is available for analysis, a placeholder character
        is inserted in the respective field.

        :raises Exception: If the method indirectly interacts with GUI widgets in an unexpected
            state, it may raise exceptions related to invalid operations or constraints.
        """

        # --- Original signal ---
        self.freq_text_orig.configure(state="normal")
        self.freq_text_orig.delete("1.0", tk.END)
        if self.audio_signal is not None:
            f_orig, X_orig = frequency_analysis(self.audio_signal, self.fs)
            top_freqs_orig, _ = get_dominant_frequencies(f_orig, np.abs(X_orig), self.n_max)
            for freq in top_freqs_orig:
                self.freq_text_orig.insert(tk.END, f"{freq:.1f}Hz\n")
        else:
            self.freq_text_orig.insert(tk.END, "—")
        self.freq_text_orig.configure(state="disabled")

        # --- Edited signal ---
        self.freq_text_mod.configure(state="normal")
        self.freq_text_mod.delete("1.0", tk.END)
        if self.audio_signal_modified is not None:
            f_mod, X_mod = frequency_analysis(self.audio_signal_modified, self.fs)
            top_freqs_mod, _ = get_dominant_frequencies(f_mod, np.abs(X_mod), self.n_max)
            for freq in top_freqs_mod:
                self.freq_text_mod.insert(tk.END, f"{freq:.1f}Hz\n")
        else:
            self.freq_text_mod.insert(tk.END, "—")
        self.freq_text_mod.configure(state="disabled")

    def update_fields(self):
        """
        Updates specific fields related to audio signal characteristics.

        This method calculates the energy and power of the `audio_signal` and
        `audio_signal_modified` attributes and updates the corresponding variables
        to reflect these calculations. Additionally, it computes and updates
        dominant frequency values based on the modified signal.

        :raises ValueError: If `audio_signal` or `audio_signal_modified` values
            are invalid, this method does not perform calculations and exits gracefully.
        """
        if self.audio_signal is None or self.audio_signal_modified is None:
            return

        # Calculating energy and power
        energy_orig, power_orig = compute_energy_power(self.audio_signal)
        energy_mod, power_mod = compute_energy_power(self.audio_signal_modified)

        # Update the lables
        self.energy_orig_var.set(f"{energy_orig:.3e}")
        self.energy_mod_var.set(f"{energy_mod:.3e}")
        self.power_orig_var.set(f"{power_orig:.3e}")
        self.power_mod_var.set(f"{power_mod:.3e}")

        self.calculate_dominant_freqs()

    def update_plot(self):
        """
        Updates the plots for the audio signals and their frequency domain representations.

        This method updates time-domain plots, spectrograms, autocorrelation, and frequency-domain representations for the
        original and modified audio signals. It supports three types of frequency-domain analysis: magnitude spectrum, phase
        spectrum, and 3-bands decomposition. The visualizations are rendered on the respective matplotlib axes.

        :param self: The class instance containing necessary attributes and methods for updating plots.
        :return: None.

        :raise ValueError: Raised if an unsupported choice is encountered in the graph selection.
        """
        if self.audio_signal_modified is None:
            return

        choice = self.graph_choice.get()

        # Clear Graphs
        self.ax_freq.clear()
        self.ax_time.clear()
        self.ax_freq2.clear()
        self.ax_time2.clear()

        # Updating time graphs
        t = np.arange(len(self.audio_signal)) / self.fs
        plot_graph(t, self.audio_signal, "Time [s]", "Amplitude", self.ax_time, self.canvas_time, "Original Signal")

        t_edited = np.arange(len(self.audio_signal_modified)) / self.fs
        plot_graph(t_edited, self.audio_signal_modified, "Time [s]", "Amplitude", self.ax_time2, self.canvas_time2,
                   "Modified Signal")

        # Updating Spectrogram and Autocorrelation
        self.calculate_spectrogram()
        self.calculate_autocorrelation()

        if choice == "magnitude":
            # Original Signal
            f, X = frequency_analysis(self.audio_signal, self.fs)
            if self.logMagnitude.get():
                X_mag = 20 * np.log10(np.abs(X) + 1e-12)
            else:
                X_mag = np.abs(X)
            plot_graph(f, X_mag, "Frequency (Hz)", "|X(f)|", self.ax_freq, self.canvas_freq, "FFT Magnitude")

            # Edited Signal
            f_edited, X_edited = frequency_analysis(self.audio_signal_modified, self.fs)
            if self.logMagnitude.get():
                X_mag_edited = 20 * np.log10(np.abs(X_edited) + 1e-12)
            else:
                X_mag_edited = np.abs(X_edited)
            plot_graph(f_edited, X_mag_edited, "Frequency (Hz)", "|X(f)|", self.ax_freq2, self.canvas_freq2,
                       "FFT Magnitude")

        elif choice == "phase":
            # Original Signal
            f, X = frequency_analysis(self.audio_signal, self.fs)
            plot_graph(f, np.angle(X), "Frequency (Hz)", "Phase (rad)", self.ax_freq, self.canvas_freq, "FFT Phase")

            # Edited Signal
            f_edited, X_edited = frequency_analysis(self.audio_signal_modified, self.fs)
            plot_graph(f_edited, np.angle(X_edited), "Frequency (Hz)", "Phase (rad)", self.ax_freq2, self.canvas_freq2,
                       "FFT Phase")

        elif choice == "3bands":
            self.graph_plot3_update()

    def graph_plot3_update(self):
        """
        Updates the plots for the frequency analysis of the original and edited audio signals,
        split into three frequency bands (low, mid, high). The function visualizes the magnitude
        spectra of these bands, either in linear or logarithmic scale, depending on configuration.

        :raises ValueError: If the audio signals are invalid or incompatible for analysis.

        :param self: Reference to the instance of the class. Contains relevant attributes
            like audio signals (`audio_signal`, `audio_signal_modified`), sampling frequency
            (`fs`), plot axes (`ax_freq`, `ax_freq2`), and canvas objects (`canvas_freq`,
            `canvas_freq2`). Additionally checks whether the log magnitude representation
            (`logMagnitude`) is enabled.
        """
        # Original Signal
        x_low, x_mid, x_high = split_into_3bands(self.audio_signal, self.fs)
        f_l, X_l = frequency_analysis(x_low, self.fs)
        f_m, X_m = frequency_analysis(x_mid, self.fs)
        f_h, X_h = frequency_analysis(x_high, self.fs)

        # Conditionally computes log magnitude for low, mid, high bands
        if self.logMagnitude.get():
            X_l_mag = 20 * np.log10(np.abs(X_l) + 1e-12)
            X_m_mag = 20 * np.log10(np.abs(X_m) + 1e-12)
            X_h_mag = 20 * np.log10(np.abs(X_h) + 1e-12)
        else:
            X_l_mag = np.abs(X_l)
            X_m_mag = np.abs(X_m)
            X_h_mag = np.abs(X_h)
        plot_3graph(f_l, X_l_mag, f_m, X_m_mag, f_h, X_h_mag, self.ax_freq, self.canvas_freq,
                    "Original Signal 3 Bands FFT", "Frequency [Hz]", "|X(f)|")

        # Edited Signal
        x_low_edited, x_mid_edited, x_high_edited = split_into_3bands(self.audio_signal_modified, self.fs)
        f_l_edited, X_l_edited = frequency_analysis(x_low_edited, self.fs)
        f_m_edited, X_m_edited = frequency_analysis(x_mid_edited, self.fs)
        f_h_edited, X_h_edited = frequency_analysis(x_high_edited, self.fs)

        # Computes magnitude of edited signal bands; logarithmic if enabled
        if self.logMagnitude.get():
            X_l_mag_edited = 20 * np.log10(np.abs(X_l_edited) + 1e-12)
            X_m_mag_edited = 20 * np.log10(np.abs(X_m_edited) + 1e-12)
            X_h_mag_edited = 20 * np.log10(np.abs(X_h_edited) + 1e-12)
        else:
            X_l_mag_edited = np.abs(X_l_edited)
            X_m_mag_edited = np.abs(X_m_edited)
            X_h_mag_edited = np.abs(X_h_edited)

        plot_3graph(f_l_edited, X_l_mag_edited, f_m_edited, X_m_mag_edited, f_h_edited, X_h_mag_edited, self.ax_freq2,
                    self.canvas_freq2, "Modified Signal 3 Bands FFT", "Frequency [Hz]", "|X(f)|")

    def load_audio(self):
        """
        Loads an audio file and initializes the audio signal and sampling frequency.

        This method allows the user to select a WAV audio file using a file dialog.
        If a file is selected, it loads the audio data and sampling frequency into
        the respective attributes. Additionally, it creates a copy of the original
        audio signal for potential modifications.

        :raises FileNotFoundError: If no file is selected or if the file cannot be found.

        :rtype: None
        """
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if path:
            self.audio_signal, self.fs = load_audio(path)
            self.audio_signal_modified = self.audio_signal.copy()

        # Update plots
        self.update_plot()
        self.update_fields()

    def play_audio(self):
        """
        Plays the modified audio signal using a separate thread for asynchronous
        audio playback. Ensures that only one instance of audio playback occurs
        at a time. If no audio signal is available, it will notify the user.

        :return: None
        """
        if self.audio_signal_modified is None:
            show_error("No signal loaded. Please load a signal first.")
            return

        # if already playing it doesn't start again...
        if self.audio_thread is not None and self.audio_thread.is_alive():
            return

        self.stop_event.clear()

        self.audio_thread = threading.Thread(target=self._play_worker, daemon=True)
        self.audio_thread.start()

    def _play_worker(self, block_size=1024):
        """
        Plays audio in a looped streaming manner with real-time control and optional
        volume adjustments. Audio is processed in blocks of a specified size. The
        method performs audio playback until stopped or an optional looping condition
        is disabled. It processes mono audio from potentially multi-channel data and
        writes the processed blocks to the output stream.

        :param block_size: The size of each audio block to be streamed. Must be an
            integer greater than zero.
        :type block_size: int
        :return: None
        """
        x = self.audio_signal_modified.astype(np.float32)

        if x.ndim == 2:
            x = x.mean(axis=1)

        fs = self.fs
        N = len(x)

        with sd.OutputStream(samplerate=fs, channels=1, dtype='float32') as stream:
            while not self.stop_event.is_set():
                i = 0
                # Streams clipped audio blocks until end or stop
                while i < N and not self.stop_event.is_set():
                    block = x[i:i + block_size] * self.volume_slider.get()

                    # Clip and conversion
                    block = np.clip(block, -1.0, 1.0).astype(np.float32)

                    stream.write(block.reshape(-1, 1))
                    i += block_size

                if not self.loop_enabled:
                    break

    def generate_sinusoid(self):
        """
        Generates a sinusoidal signal based on user-defined parameters such as frequency, amplitude, and phase.

        The method retrieves input values from user interface fields, validates them, and calculates the sinusoidal
        signal accordingly. It plots both the original and edited signals in the time and frequency domains.

        :raises ValueError: If the input values cannot be converted to float.

        :return: None
        """
        try:  # Checking if the input values are valid
            f = float(self.freq_entry.get())
            A = float(self.amp_entry.get())
            phi = float(self.phase_entry.get())
            duration = float(self.dur_entry.get())
        except ValueError:
            show_error("Invalid input value. Please enter valid numeric values.")
            return

        self.audio_signal = A * np.sin(2 * np.pi * f * np.arange(0, duration, 1 / self.fs) + phi)
        self.audio_signal_modified = self.audio_signal.copy()

        # Update plots
        self.update_plot()

    def stop_audio(self):
        """
        Stops the audio playback by setting the stop event.

        This method allows signaling to stop audio playback by manipulating the
        `stop_event`. Once triggered, audio playback should cease based on external
        handling of the event.

        :return: None
        """
        self.stop_event.set()

    def loop_play_audio(self):
        """
        Toggles the looping functionality for audio playback. The method enables or disables the
        looping mode of audio playback and updates the corresponding button's label text to reflect
        the current state.

        :return: None
        """
        self.loop_enabled = not self.loop_enabled
        self.loop_button.config(text="Loop ON" if self.loop_enabled else "Loop OFF")

    def save_audio_file(self):
        """
        Saves the modified audio signal to a user-specified file location.

        This method opens a file dialog allowing the user to choose a file path and name for saving
        the modified audio signal. The file is stored in the WAV file format. If the user cancels
        the save operation or does not choose a file path, the method exits without saving.

        :raises FileNotFoundError: If the file path is invalid or inaccessible.
        :raises PermissionError: If there is not enough permission to write to the selected file path.

        :return: None
        """
        # Ask where to save the file
        path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if not path:
            return

        # Save the edited audio
        save_audio(path, self.audio_signal_modified, self.fs)

    def save_params(self):  # TODO
        pass

    def load_params(self):  # TODO
        pass

    def apply_processing(self):
        """
        Process the audio signal with user-defined equalization and modulation settings, update the modified audio
        signal, and plot the results in both time and frequency domains.

        The method first ensures there is an available audio signal that can be processed. If equalization or
        modulation is enabled, it applies those processing steps using the parameters set by the user. Finally,
        the processed signal replaces the original modified signal, and the results are visualized in the
        corresponding time and frequency domain plots.

        :param self: The instance of the class calling this method.

        :raises ValueError: If the given input parameters or configurations are invalid.
        :return: None
        """
        if self.audio_signal is None:
            show_error("No signal loaded. Please load a signal first.")
            return

        processed_signal = self.audio_signal.copy()

        if self.equalizer_enabled.get():
            processed_signal = equalize(processed_signal, self.low_gain.get(), self.mid_gain.get(),
                                        self.high_gain.get(), self.fs)

        if self.modulation_enabled.get():
            processed_signal = modulate_signal(processed_signal, self.fs, self.fc_slider.get(), self.k_slider.get(),
                                               self.mod_type_var.get())
        self.audio_signal_modified = processed_signal

        # Update plots
        self.update_plot()
        self.update_fields()

    def on_volume_change(self, v):
        """
        Handles volume change event by updating the label text to display the new volume
        value, formatted to two decimal places. If the input value cannot be converted
        to a float, the method safely exits without updating.

        :param v: The new volume value as a string. The method attempts to convert this
            value to a float.
        :return: None
        """
        try:
            v_float = float(v)
        except ValueError:
            return
        self.volume_value_label.config(text=f"{v_float:.2f}")
