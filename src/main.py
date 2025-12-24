import numpy as np

from src.logic.IO import load_audio, to_mono
from src.logic.analysis import frequency_analysis, create_time_plot, create_frequency_plot, get_dominant_frequencies, \
    compute_energy_power, check_Parseval_theorem, spectrogram_analysis, plot_fft_3d, autocorrelation_plot

from src.logic.equalizer import split_into_bands, equalize
from src.logic.modulation import modulate_signal
from src.logic.utils import plot_graph, plot_two_graphs_side_by_side, plot_bands_time, plot_bands_frequency


def pipeline_test():
    # Acquisition of the signal
    x_in, fs = load_audio("./../data/LA_Piano.wav")
    print(f"Sample rate: {fs} Hz")

    # Ensure mono
    x = to_mono(x_in)

    # Plot time domain
    plot_graph(create_time_plot(x, fs))

    # Plot 3D graph of DFT
    plot_graph(plot_fft_3d(x, fs))

    # Plot frequency domain
    f, X_mag = frequency_analysis(x, fs)
    plot_graph(create_frequency_plot(X_mag, f))

    # Show dominant frequency
    print(get_dominant_frequencies(f, X_mag, 10))

    # Calculating power and energy and verifying Parseval Theorem
    energy, power = compute_energy_power(x)
    print(energy, power)
    print(check_Parseval_theorem(x, np.fft.fft(x)))

    # Autocorrelation Calculation
    plot_graph(autocorrelation_plot(x))

    # Splitting signal in bands
    x_low, x_mid, x_high = split_into_bands(x, fs)
    plot_bands_time(x_low, x_mid, x_high, fs)
    f_l, X_l = frequency_analysis(x_low, fs)
    f_m, X_m = frequency_analysis(x_mid, fs)
    f_h, X_h = frequency_analysis(x_high, fs)
    plot_bands_frequency(f_l, X_l, f_m, X_m, f_h, X_h, fs)

    equalized_signal = equalize(x, 0.8, 1.2, 1.5, fs)

    t = np.arange(len(x)) / fs
    t_eq = np.arange(len(equalized_signal)) / fs

    plot_two_graphs_side_by_side(
        t, x,
        "Original signal",
        "Time [s]",
        "Amplitude",

        t_eq, equalized_signal,
        "Equalized signal",
        "Time [s]",
        "Amplitude"
    )

    # Signal modulation
    x_am = modulate_signal(equalized_signal, fs, 5000, k=0.5, mod_type='AM')

    # Stft and visualization
    plot_graph(spectrogram_analysis(x_am, fs))

    # Saving/Reproducing signal

def main():
    # pipeline_test()
    from src.gui.gui import AudioDSPApp
    app = AudioDSPApp()
    app.mainloop()
    pass


if __name__ == "__main__":
    main()
