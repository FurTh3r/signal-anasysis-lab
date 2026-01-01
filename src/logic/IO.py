import numpy as np
import sounddevice as sd
import soundfile as sf


def load_audio(file_path: str) -> tuple[np.ndarray, int]:
    """
    Loads an audio file from the given file path, converts the audio data to
    32-bit float type if necessary, and ensures the audio data is converted
    to mono channel format.

    :param file_path: The path to the audio file to be loaded.
    :type file_path: str
    :return: A tuple containing the audio data as a NumPy array and the
        sampling rate as an integer.
    :rtype: tuple[np.ndarray, int]
    """
    x, fs = sf.read(file_path)
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    return to_mono(x), fs


def to_mono(x: np.ndarray) -> np.ndarray:
    """
    Convert a multidimensional audio signal array into a mono-dimensional signal.

    This function processes audio signals represented as NumPy arrays. If the input
    is already a 1D array, it is returned as is. If the input is 2D, the function
    computes the mean across the second dimension (axis=1) to convert it to mono.
    For inputs of dimensions other than 1 or 2, a ValueError is raised.

    :param x: The input audio signal. Should be either a 1D or 2D NumPy array.
    :type x: np.ndarray
    :return: A mono-dimensional NumPy array representing the audio signal.
    :rtype: np.ndarray
    :raises ValueError: If the input array does not have 1D or 2D dimensions.
    """
    if x.ndim == 1:
        return x
    elif x.ndim == 2:
        mono_x = x.mean(axis=1)
        return mono_x
    else:
        raise ValueError("Input audio must be 1D or 2D array.")


def save_audio(file_path: str, x: np.ndarray, fs: int):
    """
    Saves an audio file to the specified path. The audio data is provided as a NumPy
    array, and the sampling frequency is specified as an integer.

    :param file_path: The path where the audio file will be saved.
    :type file_path: str
    :param x: A NumPy array containing the audio data to be saved.
    :type x: numpy.ndarray
    :param fs: The sampling frequency of the audio data.
    :type fs: int
    :return: None
    """
    sf.write(file_path, x, fs)


def reproduce_audio(x: np.ndarray, fs: int):
    """
    Reproduces audio data using a given sampling frequency.

    This function uses the sounddevice library to play audio data stored in a
    NumPy array. The function will block execution until the audio playback
    is finished. The audio data should be provided as an array, while its
    sampling frequency should be provided as an integer value.

    :param x: A NumPy array containing the audio data to be played.
    :param fs: An integer representing the sampling frequency of the
        audio data in Hz.
    :return: None
    """
    sd.play(x, samplerate=fs)
    sd.wait()
