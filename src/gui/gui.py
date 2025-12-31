import threading
import tkinter as tk
from tkinter import ttk, filedialog

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

from src.logic.IO import load_audio, record_audio, save_audio
from src.logic.equalizer import equalize
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


class AudioDSPApp(tk.Tk):
    """
    Main application class for a graphical user interface (GUI) toolkit designed for audio DSP (Digital Signal Processing).
    This class inherits from `tk.Tk` and serves as the entry point for creating and managing the GUI components, including
    audio control and signal analysis features.
    """

    def __init__(self):
        """
        Initializes the main application for the Audio DSP Toolkit.
        """
        super().__init__()

        self.title("Audio DSP Toolkit")
        self.geometry("1200x800")
        self.configure(bg=BG_MAIN)

        # Component Variables:
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

        self.tab_control.add(self.tab_control_tab1, text='Elaborazione Audio')
        self.tab_control.add(self.tab_control_tab2, text='Analisi del Segnale')

        self.create_signal_analysis_tab()
        self.create_audio_control_tab()

    # =========================
    # TAB 1 – Audio Control
    # =========================
    def create_audio_control_tab(self):
        # =======================
        # Layout principale
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

        # Volume
        vol_frame = ttk.Frame(frame_io)
        vol_frame.pack(side="left", padx=15)
        ttk.Label(vol_frame, text="Volume").pack()
        self.volume_value_label = ttk.Label(vol_frame, text="1.00")
        self.volume_value_label.pack()

        self.volume_slider = tk.Scale(vol_frame, from_=1.0, to=0.0, resolution=0.01, orient="vertical", length=120,
                                      command=lambda v: self.volume_value_label.config(text=f"{float(v):.2f}"))
        self.volume_slider.set(1.0)
        self.volume_slider.pack()

        # =======================
        # Equalizer
        frame_eq = ttk.Labelframe(left_frame, text="Equalizzatore")
        frame_eq.pack(fill="x", pady=10)

        self.low_gain = _create_vertical_slider(frame_eq, "Low", 1.0)
        self.mid_gain = _create_vertical_slider(frame_eq, "Mid", 1.0)
        self.high_gain = _create_vertical_slider(frame_eq, "High", 1.0)
        ttk.Checkbutton(frame_eq, text="Enable EQ", variable=self.equalizer_enabled).pack(side="left", padx=10)

        # =======================
        # Modulation
        frame_mod = ttk.Labelframe(left_frame, text="Modulazione")
        frame_mod.pack(fill="x", pady=10)

        self.mod_type_var = tk.StringVar(value="AM")
        ttk.Label(frame_mod, text="Tipo").pack(side="left")
        ttk.OptionMenu(frame_mod, self.mod_type_var, "AM", "AM", "DSB-SC", "FM").pack(side="left", padx=5)

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

        # ===== TIME GRAPH =====
        self.fig_time = plt.Figure(figsize=(4, 3))
        self.ax_time = self.fig_time.add_subplot(111)

        self.canvas_time = FigureCanvasTkAgg(self.fig_time, master=graphs_frame)
        self.canvas_time.get_tk_widget().grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        toolbar_time = NavigationToolbar2Tk(self.canvas_time, graphs_frame, pack_toolbar=False)
        toolbar_time.update()
        toolbar_time.grid(row=1, column=0, sticky="w")

        # ===== FREQUENCY GRAPH =====
        self.fig_freq = plt.Figure(figsize=(4, 3))
        self.ax_freq = self.fig_freq.add_subplot(111)

        self.canvas_freq = FigureCanvasTkAgg(self.fig_freq, master=graphs_frame)
        self.canvas_freq.get_tk_widget().grid(row=2, column=0, padx=5, pady=5, sticky="nsew")

        toolbar_freq = NavigationToolbar2Tk(self.canvas_freq, graphs_frame, pack_toolbar=False)
        toolbar_freq.update()
        toolbar_freq.grid(row=3, column=0, sticky="w")

    # =========================
    # TAB 2 – Signal Analysis
    # =========================
    def create_signal_analysis_tab(self):
        ttk.Label(self.tab_control_tab2, text="Analisi avanzata (DFT 3D, spettrogrammi, autocorrelazione)").pack(
            pady=20)

    # =========================
    # Audio GUI logic
    # =========================

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

    def play_audio(self):
        """
        Plays the modified audio signal using a separate thread for asynchronous
        audio playback. Ensures that only one instance of audio playback occurs
        at a time. If no audio signal is available, it will notify the user.

        :return: None
        """
        if self.audio_signal_modified is None:
            print("No signal to play")
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
        :raises PermissionError: If there is insufficient permission to write to the selected file path.

        :return: None
        """
        # Ask where to save the file
        path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if not path:
            return

        # Save the editedAudio
        save_audio(path, self.audio_signal_modified, self.fs)

    def save_params(self):
        pass

    def load_params(self):
        pass

    def apply_processing(self):
        if self.audio_signal_modified is None:
            print("No signal to equalize")
            return

        equalized_signal = self.audio_signal_modified.copy()
        if self.equalizer_enabled.get():
            equalized_signal = equalize(self.audio_signal, self.low_gain.get(), self.mid_gain.get(),
                                        self.high_gain.get(), self.fs)

        if self.modulation_enabled.get():
            self.audio_signal_modified = modulate_signal(equalized_signal, self.fs, self.fc_slider.get(),
                                                         self.k_slider.get(), self.mod_type_var.get())

        self.plot_time()
        self.plot_frequency()

    def plot_time(self):
        """
        Plots the time-domain representation of the modified audio signal.

        This method visualizes the audio signal in the time domain using the assigned
        axes object. It clears any previous content in the axes, computes the time
        vector based on the sampling frequency, and updates the plot with the new data.
        The method also customizes the plot's appearance, including setting labels,
        grid, and title.

        :return: None
        """
        if self.audio_signal_modified is None or self.ax_time is None:
            return

        self.ax_time.clear()
        t = np.arange(len(self.audio_signal_modified)) / self.fs
        self.ax_time.set_xlim(0, t[-1])
        self.ax_time.set_ylim(1.1 * np.min(self.audio_signal_modified), 1.1 * np.max(self.audio_signal_modified))
        self.ax_time.plot(t, self.audio_signal_modified)
        self.ax_time.set_title("Segnale nel tempo")
        self.ax_time.set_xlabel("Tempo [s]")
        self.ax_time.set_ylabel("Ampiezza")
        self.ax_time.grid(True)
        self.canvas_time.draw_idle()

    def plot_frequency(self):
        """
        Plots the frequency spectrum of a modified audio signal.

        This method calculates and plots the frequency components of the modified audio
        signal. It uses the Fast Fourier Transform (FFT) to compute the frequency
        spectrum and plots it on the corresponding subplot axis. The x-axis represents
        the frequency in Hertz, while the y-axis represents the magnitude of the
        frequency components. The method updates the plot dynamically if the canvas is
        available.

        :raises ValueError: If either the `audio_signal_modified` or `ax_freq` attribute
            is not set, the method will not proceed and will return without producing
            any plot.

        :return: None
        """
        if self.audio_signal_modified is None or self.ax_freq is None:
            return

        self.ax_freq.clear()
        X = np.fft.rfft(self.audio_signal_modified)
        f = np.fft.rfftfreq(len(self.audio_signal_modified), 1 / self.fs)
        self.ax_freq.set_xlim(0, self.fs / 2)
        self.ax_freq.plot(f, np.abs(X))
        self.ax_freq.set_title("Spettro in frequenza")
        self.ax_freq.set_xlabel("Frequenza [Hz]")
        self.ax_freq.set_ylabel("|X(f)|")
        self.ax_freq.grid(True)
        self.canvas_freq.draw_idle()
