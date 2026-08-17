# Audio DSP Toolkit

A Python desktop application for exploring digital signal-processing techniques on audio signals. The project combines waveform inspection, Fourier analysis, filtering, equalization, modulation, playback, and visualization in a Tkinter interface.

## Features

- load and save audio files;
- generate configurable sinusoidal signals;
- play audio with volume and looping controls;
- display time-domain waveforms and frequency spectra;
- identify dominant frequency components;
- compute signal energy and average power;
- verify Parseval's theorem numerically;
- compute autocorrelation and short-time Fourier transforms;
- display spectrograms and 3D FFT visualizations;
- split signals into low-, mid-, and high-frequency bands;
- apply three-band equalization;
- apply AM and FM modulation;
- compare original and processed signals.

Sample WAV files are included under data/.

## Requirements

- Python 3.13, as used by the pinned dependency set;
- Tk support for Python;
- PortAudio or an equivalent backend required by sounddevice;
- the packages listed in requirements.txt.

On Debian or Ubuntu, the native GUI/audio prerequisites can typically be installed with:

    sudo apt install python3-tk libportaudio2

## Installation

Create and activate a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate

On Windows PowerShell:

    .venv\Scripts\Activate.ps1

Install the Python dependencies:

    python -m pip install --upgrade pip
    pip install -r requirements.txt

## Running the Application

Run the module from the repository root so that src imports resolve correctly:

    python -m src.main

The application starts in GUI mode by default.

## Programmatic Pipeline

src/main.py also contains pipeline_test(), a scripted demonstration of the processing stages. To use it, set application to False in main() and verify the audio_file path before running the module.

The pipeline covers:

1. audio loading;
2. time-domain plotting;
3. FFT analysis;
4. dominant-frequency extraction;
5. energy, power, and Parseval verification;
6. autocorrelation;
7. band splitting and equalization;
8. AM or FM modulation;
9. STFT and spectrogram generation.

## Repository Structure

    .
    ├── data/                  # Sample WAV signals
    ├── requirements.txt
    └── src/
        ├── main.py            # Application entry point and test pipeline
        ├── gui/
        │   └── gui.py         # Tkinter interface
        └── logic/
            ├── IO.py          # Audio input and output
            ├── analysis.py    # Spectral and statistical analysis
            ├── equalizer.py   # Band splitting and equalization
            ├── filters.py     # Digital filters
            ├── modulation.py  # AM and FM modulation
            └── utils.py       # Plotting and helper functions

## Notes

- Audio playback depends on the devices exposed by sounddevice.
- GUI execution requires a graphical desktop session.
- The repository name preserves the original spelling, signal-anasysis-lab.

## Author

**Lorenzo Pasini** — [FurTh3r](https://github.com/FurTh3r)

## License

No license file is currently included in this repository. Unless otherwise stated by the author, all rights are reserved.
