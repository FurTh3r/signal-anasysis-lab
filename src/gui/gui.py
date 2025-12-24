import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src.logic.IO import load_audio, reproduce_audio, save_audio, record_audio
import numpy as np

# =========================
# Palette colori
# =========================
BG_MAIN = "#eaf4fb"
BG_FRAME = "#dbeeff"
ACCENT = "#4aa3df"
ACCENT_DARK = "#2b7bb9"
TEXT_COLOR = "#1f2d3d"


class AudioDSPApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Audio DSP Toolkit")
        self.geometry("1200x800")
        self.configure(bg=BG_MAIN)

        # =========================
        # Stile moderno
        # =========================
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("TFrame", background=BG_MAIN)
        style.configure("TLabelframe", background=BG_FRAME)
        style.configure("TLabelframe.Label",
                        background=BG_FRAME,
                        foreground=TEXT_COLOR,
                        font=("Segoe UI", 10, "bold"))

        style.configure("TButton",
                        background=ACCENT,
                        foreground="white",
                        padding=6,
                        font=("Segoe UI", 9))
        style.map("TButton",
                  background=[("active", ACCENT_DARK)])

        style.configure("TNotebook", background=BG_MAIN, borderwidth=0)
        style.configure("TNotebook.Tab",
                        background=BG_FRAME,
                        foreground=TEXT_COLOR,
                        padding=(12, 6))
        style.map("TNotebook.Tab",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", "white")])

        # =========================
        # Variabili audio
        # =========================
        self.audio_signal = None
        self.fs = 44100

        # =========================
        # Notebook
        # =========================
        self.tab_control = ttk.Notebook(self)
        self.tab_control.pack(expand=1, fill="both")

        self.tab_control_tab1 = ttk.Frame(self.tab_control)
        self.tab_control_tab2 = ttk.Frame(self.tab_control)

        self.tab_control.add(self.tab_control_tab1, text='Elaborazione Audio')
        self.tab_control.add(self.tab_control_tab2, text='Analisi del Segnale')

        self.create_audio_control_tab()
        self.create_signal_analysis_tab()

    # =========================
    # TAB 1 – Audio Control
    # =========================
    def create_audio_control_tab(self):
        # =======================
        # Controlli audio
        # =======================
        frame_io = ttk.Labelframe(self.tab_control_tab1, text="Audio I/O")
        frame_io.pack(fill="x", padx=15, pady=10)

        # Prima riga: Load e Save
        top_frame = ttk.Frame(frame_io)
        top_frame.pack(fill="x", pady=5)
        ttk.Button(top_frame, text="Load", command=self.load_audio).pack(side="left", padx=5)
        ttk.Button(top_frame, text="Save", command=self.save_audio).pack(side="left", padx=5)

        # Seconda riga: Record, Play, Loop
        bottom_frame = ttk.Frame(frame_io)
        bottom_frame.pack(fill="x", pady=5)
        ttk.Button(bottom_frame, text="Record", command=self.record_audio).pack(side="left", padx=5)
        ttk.Button(bottom_frame, text="Play", command=self.play_audio).pack(side="left", padx=5)
        ttk.Button(bottom_frame, text="Loop", command=self.loop_play_audio).pack(side="left", padx=5)

        # Volume verticale
        vol_frame = ttk.Frame(frame_io)
        vol_frame.pack(side="left", padx=20)
        ttk.Label(vol_frame, text="Volume").pack()
        self.volume_value_label = ttk.Label(vol_frame, text="1.00")
        self.volume_value_label.pack()
        self.volume_slider = tk.Scale(vol_frame, from_=1.0, to=0.0, resolution=0.01,
                                      orient="vertical", length=150,
                                      command=lambda v: self.volume_value_label.config(text=f"{float(v):.2f}"))
        self.volume_slider.set(1.0)
        self.volume_slider.pack()

        # =======================
        # Equalizzatore verticale
        # =======================
        frame_eq = ttk.Labelframe(self.tab_control_tab1, text="Equalizzatore")
        frame_eq.pack(fill="x", padx=15, pady=10)

        self.low_gain = self._create_vertical_slider(frame_eq, "Low", 1.0)
        self.mid_gain = self._create_vertical_slider(frame_eq, "Mid", 1.0)
        self.high_gain = self._create_vertical_slider(frame_eq, "High", 1.0)

        ttk.Button(frame_eq, text="Apply EQ", command=self.apply_equalizer).pack(side="left", padx=10)

        # =======================
        # Modulazione (orizzontale)
        # =======================
        frame_mod = ttk.Labelframe(self.tab_control_tab1, text="Modulazione")
        frame_mod.pack(fill="x", padx=15, pady=10)

        self.mod_type_var = tk.StringVar(value="AM")
        ttk.Label(frame_mod, text="Tipo").pack(side="left")
        ttk.OptionMenu(frame_mod, self.mod_type_var, "AM", "AM", "DSB-SC", "FM").pack(side="left", padx=5)

        ttk.Label(frame_mod, text="f_c [Hz]").pack(side="left", padx=(10, 2))
        self.fc_slider = tk.Scale(frame_mod, from_=100, to=20000, orient="horizontal",
                                  length=200, resolution=1)
        self.fc_slider.set(5000)
        self.fc_slider.pack(side="left", fill="x", expand=True)

        ttk.Label(frame_mod, text="k").pack(side="left", padx=(10, 2))
        self.k_slider = tk.Scale(frame_mod, from_=0.0, to=5.0, orient="horizontal",
                                 length=150, resolution=0.01)
        self.k_slider.set(0.5)
        self.k_slider.pack(side="left", fill="x", expand=True)

        ttk.Button(frame_mod, text="Apply Modulation", command=self.apply_modulation).pack(side="left", padx=10)

    # =========================
    # Helper per slider verticali
    # =========================
    def _create_vertical_slider(self, parent, label, default):
        frame = ttk.Frame(parent)
        frame.pack(side="left", padx=15)

        ttk.Label(frame, text=label).pack()
        value_label = ttk.Label(frame, text=f"{default:.2f}")
        value_label.pack()

        slider = tk.Scale(frame, from_=2.0, to=0.0, resolution=0.01, orient="vertical",
                          length=150, command=lambda v: value_label.config(text=f"{float(v):.2f}"))
        slider.set(default)
        slider.pack()
        return slider

    # =========================
    # TAB 2 – Signal Analysis
    # =========================
    def create_signal_analysis_tab(self):
        frame_graphs = ttk.Frame(self.tab_control_tab2)
        frame_graphs.pack(expand=True, fill="both", padx=10, pady=10)

        self.fig_time = plt.Figure(figsize=(4, 3), facecolor=BG_FRAME)
        self.fig_freq = plt.Figure(figsize=(4, 3), facecolor=BG_FRAME)
        self.fig_3d = plt.Figure(figsize=(4, 3), facecolor=BG_FRAME)
        self.fig_spec = plt.Figure(figsize=(4, 3), facecolor=BG_FRAME)

        self.canvas_time = FigureCanvasTkAgg(self.fig_time, master=frame_graphs)
        self.canvas_freq = FigureCanvasTkAgg(self.fig_freq, master=frame_graphs)
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=frame_graphs)
        self.canvas_spec = FigureCanvasTkAgg(self.fig_spec, master=frame_graphs)

        self.canvas_time.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)
        self.canvas_freq.get_tk_widget().grid(row=0, column=1, padx=5, pady=5)
        self.canvas_3d.get_tk_widget().grid(row=1, column=0, padx=5, pady=5)
        self.canvas_spec.get_tk_widget().grid(row=1, column=1, padx=5, pady=5)

        self.add_graph_buttons(frame_graphs, 0, 0, self.canvas_time)
        self.add_graph_buttons(frame_graphs, 0, 1, self.canvas_freq)
        self.add_graph_buttons(frame_graphs, 1, 0, self.canvas_3d)
        self.add_graph_buttons(frame_graphs, 1, 1, self.canvas_spec)

    def apply_modulation(self):
        if self.audio_signal is None:
            return

        m = self.audio_signal
        fs = self.fs
        n = np.arange(len(m))

        fc = self.fc_slider.get()
        k = self.k_slider.get()
        wc = 2 * np.pi * fc / fs

        mod_type = self.mod_type_var.get()

        if mod_type == "AM":
            y = (1 + k * m) * np.cos(wc * n)

        elif mod_type == "DSB-SC":
            y = m * np.cos(wc * n)

        elif mod_type == "FM":
            phase = wc * n + k * np.cumsum(m)
            y = np.cos(phase)

        self.audio_signal = y / np.max(np.abs(y))

    # =========================
    # Audio logic (placeholder)
    # =========================
    def load_audio(self):
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if path:
            self.audio_signal, self.fs = load_audio(path)

    def record_audio(self):
        self.audio_signal = record_audio(duration=5, fs=self.fs)

    def play_audio(self):
        if self.audio_signal is not None:
            reproduce_audio(self.audio_signal, self.fs)

    def loop_play_audio(self):
        pass

    def save_audio(self):
        pass

    def apply_equalizer(self):
        pass

    def save_params(self):
        pass

    def load_params(self):
        pass

    # =========================
    # Graph controls
    # =========================
    def add_graph_buttons(self, parent, row, column, canvas):
        frame = ttk.Frame(parent)
        frame.grid(row=row + 1, column=column, pady=2)

        ttk.Button(frame, text="⤢", width=3).pack(side="left")
        ttk.Button(frame, text="▢", width=3).pack(side="left")
        ttk.Button(frame, text="💾", width=3).pack(side="left")