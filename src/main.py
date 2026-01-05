import numpy as np

from src.logic.IO import load_audio
from src.logic.analysis import frequency_analysis, create_time_plot, create_frequency_plot, get_dominant_frequencies, \
    compute_energy_power, check_parseval_theorem, spectrogram_analysis, plot_fft_3d, autocorrelation_plot
from src.logic.equalizer import split_into_3bands, equalize
from src.logic.modulation import modulate_signal
from src.logic.utils import plot_graph, plot_two_graphs_side_by_side, plot_bands_time, plot_bands_frequency


def pipeline_test():
    """
    Steps:
    1. Load signal
    2. Time-domain plot
    3. 3D FFT plot
    4. Frequency analysis
    5. Dominant frequencies
    6. Energy & Power + Parseval check
    7. Autocorrelation
    8. Band splitting & plots
    9. Equalization
    10. Signal modulation (AM or FM)
    11. STFT / spectrogram
    """

    # --------------------------
    # PARAMETERS
    # --------------------------
    audio_file = "../data/logChirpUp.wav"  # path to audio
    modulation_type = 'FM'  # 'AM' or 'FM'

    # Flags to enable/disable processing steps
    modulation = False  # True = apply modulation, False = skip
    equalization = False  # True = apply equalization, False = skip

    # Carrier parameters
    carrier_freq = 5000  # Hz
    modulation_index = 0.5  # AM: amplitude index, FM: frequency deviation factor

    # Equalizer gains
    equalizer_low = 0.8
    equalizer_mid = 1.2
    equalizer_high = 1.5

    num_dominant_freqs = 10

    # SIGNAL ACQUISITION
    x, fs = load_audio(audio_file)
    print(f"Sample rate: {fs} Hz")

    # TIME DOMAIN PLOT
    plot_graph(create_time_plot(x, fs))

    # 3D FFT PLOT
    plot_graph(plot_fft_3d(x, fs))

    # FREQUENCY ANALYSIS
    f, X = frequency_analysis(x, fs)
    plot_graph(create_frequency_plot(np.abs(X), f))

    # DOMINANT FREQUENCIES
    dominant_freqs = get_dominant_frequencies(f, np.abs(X), num_dominant_freqs)
    print(f"Top {num_dominant_freqs} dominant frequencies: {dominant_freqs}")

    # ENERGY & POWER + PARSEVAL CHECK
    energy, power = compute_energy_power(x)
    print(f"Signal energy: {energy}, power: {power}")
    parseval_check = check_parseval_theorem(x, np.fft.fft(x))
    print(f"Parseval theorem verified: {parseval_check}")

    # AUTOCORRELATION
    plot_graph(autocorrelation_plot(x))

    # SPLIT INTO BANDS
    x_low, x_mid, x_high = split_into_3bands(x, fs)
    plot_bands_time(x_low, x_mid, x_high, fs)
    f_l, X_l = frequency_analysis(x_low, fs)
    f_m, X_m = frequency_analysis(x_mid, fs)
    f_h, X_h = frequency_analysis(x_high, fs)
    plot_bands_frequency(f_l, np.abs(X_l), f_m, np.abs(X_m), f_h, np.abs(X_h))

    # EQUALIZATION
    if equalization:
        equalized_signal = equalize(x, equalizer_low, equalizer_mid, equalizer_high, fs)
    else:
        equalized_signal = x.copy()

    t = np.arange(len(x)) / fs
    t_eq = np.arange(len(equalized_signal)) / fs

    plot_two_graphs_side_by_side(t, x, "Original signal", "Time [s]", "Amplitude", t_eq, equalized_signal,
                                 "Equalized signal", "Time [s]", "Amplitude")

    # AM-FM MODULATION
    if modulation:
        modulated_signal = modulate_signal(equalized_signal, fs, fc=carrier_freq, k=modulation_index,
                                           mod_type=modulation_type)
    else:
        modulated_signal = equalized_signal.copy()

    # STFT / SPECTROGRAM
    plot_graph(spectrogram_analysis(modulated_signal, fs))

    print(f"Pipeline execution completed successfully with {modulation_type.upper()} modulation.")


def application_start():
    from src.gui.gui import AudioDSPApp
    app = AudioDSPApp()
    app.mainloop()
    pass


def main():
    application = True

    if application:
        application_start()
    else:
        pipeline_test()


if __name__ == "__main__":
    main()
