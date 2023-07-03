import os
import pickle

import librosa
import numpy as np

from Pipeline.filtering import butter_bandpass_filter


class Loader:
    """Loader is responsible for loading an audio file."""

    def __init__(self, sample_rate, duration=None, mono=True):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        signal, _ = librosa.load(file_path,
                                 sr=self.sample_rate,
                                 duration=self.duration,
                                 mono=self.mono)
        return signal


class BandpassFilter:
    """BandpassFilter is responsible for applying a bandpass filter to an array."""

    def __init__(self, low_cut, high_cut, sample_rate, order=2):
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.sample_rate = sample_rate
        self.order = order

    def filter(self, signal):
        signal = butter_bandpass_filter(signal, self.low_cut, self.high_cut, self.sample_rate, self.order)
        return signal


class Padder:
    """Padder is responsible to apply padding to an array."""

    def __init__(self, mode="constant"):
        self.mode = mode

    def left_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (num_missing_items, 0),
                              mode=self.mode)
        return padded_array

    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (0, num_missing_items),
                              mode=self.mode)
        return padded_array


class SpectrogramExtractor:
    """SpectrogramExtractor extracts spectrograms from a time-series signal."""

    def __init__(self, n_fft, win_length, hop_length, mode="spectrogram"):
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.mode = mode

    def extract_spectrogram(self, signal, log=False):
        stft = librosa.stft(
            y=signal,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window='hann',
            center=False
        )
        spectrogram = np.abs(stft)
        if log:
            spectrogram = librosa.amplitude_to_db(spectrogram)
        return spectrogram

    def extract_mel_spectrogram(self, signal, sample_rate, log=False, power=2.0, n_mels=64):
        mel_spec = librosa.feature.melspectrogram(
            y=signal,
            sr=sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window='hann',
            center=False,
            power=power,  # 1 for energy, 2 for power
            n_mels=n_mels,
        )
        spectrogram = np.abs(mel_spec)
        if log:
            if power == 1.0:
                spectrogram = librosa.amplitude_to_db(spectrogram)
            else:
                spectrogram = librosa.power_to_db(spectrogram)
        return spectrogram

    def extract_cqt(self, signal, sample_rate, log=False):
        cqt = librosa.cqt(
            y=signal,
            sr=sample_rate,
            n_bins=self.n_fft // 2 + 1,
            hop_length=self.hop_length,
            window='hann',
        )
        spectrogram = np.abs(cqt)
        if log:
            spectrogram = librosa.amplitude_to_db(spectrogram)
        return spectrogram

    def extract(self, signal, sample_rate, log=False):
        if self.mode == "spectrogram":
            return self.extract_spectrogram(signal, log=log)
        elif self.mode == "mel_spectrogram":
            return self.extract_mel_spectrogram(signal, sample_rate, log=log)
        elif self.mode == "cqt":
            return self.extract_cqt(signal, sample_rate, log=log)
        else:
            raise ValueError("Spectrogram mode not supported")


class Normalizer:
    """Normalizer applies normalization to an array."""

    def __init__(self, mode="none"):
        self.mode = mode

    @staticmethod
    def min_max_normalize(array, max_value=1, min_value=0):
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (max_value - min_value) + min_value
        return norm_array

    @staticmethod
    def min_max_denormalize(norm_array, original_min, original_max, max_value=1, min_value=0):
        array = (norm_array - min_value) / (max_value - min_value)
        array = array * (original_max - original_min) + original_min
        return array

    @staticmethod
    def mean_std_normalize(array):
        norm_array = (array - array.mean()) / array.std()
        return norm_array

    @staticmethod
    def unit_peak_normalize(array):
        norm_array = librosa.util.normalize(array)
        return norm_array

    def normalize(self, array):
        if self.mode == "min_max":
            return self.min_max_normalize(array)
        elif self.mode == "mean_std":
            return self.mean_std_normalize(array)
        elif self.mode == "unit_peak":
            return self.unit_peak_normalize(array)
        elif self.mode == "none":
            return array
        else:
            raise ValueError("Normalization mode not supported")


class FrequencySelector:
    """FrequencySelector is responsible to select a frequency range from a spectrogram."""

    def __init__(self, min_freq, max_freq):
        self.min_freq = min_freq
        self.max_freq = max_freq

    def select(self, spectrogram, n_fft, sample_rate):
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        in_band = np.where([self.min_freq <= f <= self.max_freq for f in freqs])[0]
        return spectrogram[in_band, :], freqs[in_band]


class SpectrogramSegmenter:
    """SpectrogramSegmenter is responsible to segment a spectrogram
    in chucks of segment_size (given in seconds) with segment_hop_size
    (given in seconds) overlap (i.e. hopping window)"""

    def __init__(self, segment_size, segment_hop_size, sample_rate, hop_length):
        if segment_hop_size > segment_size:
            raise ValueError('segment_hop_size must be smaller than segment_size')

        self.chunk_win = int(segment_size * sample_rate / hop_length)
        self.chunk_hop = int(segment_hop_size * sample_rate / hop_length)

    def segment(self, spectrogram):
        splits = []
        start = 0
        fin = self.chunk_win

        while fin < len(spectrogram[1]):
            # do not save the last chunk if it is smaller than chunk_win
            if fin - start < self.chunk_win:
                break
            splits.append(spectrogram[:, start:fin])
            start = fin - self.chunk_hop
            fin = start + self.chunk_win

        return splits

    def get_chunk(self, frames):
        """
        Given a number of frames from the original spectrogram,
        compute the corresponding chunk
        """
        chunk = 0
        fin = self.chunk_win

        while fin < frames:
            chunk += 1
            start = fin - self.chunk_hop
            fin = start + self.chunk_win

        return chunk


class Saver:
    """Saver is responsible to save the spectrogram."""

    def __init__(self, feature_save_dir):
        self.feature_save_dir = feature_save_dir

    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)

    @staticmethod
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path


class PreprocessingPipeline:
    """PreprocessingPipeline processes audio files in a directory, applying
    the following steps to each file:
        1- load a file
        2- pad the signal (if necessary)
        3- apply bandpass filter
        4- extracting log spectrogram from signal
        5- normalize spectrogram
        6- select frequency range
        7- segment spectrogram
        8- save the segmented spectrograms
    """

    def __init__(self):
        self.padder = None
        self.bandpass = None
        self.extractor = None
        self.normalizer = None
        self.selector = None
        self.segmenter = None
        self.saver = None
        self._loader = None
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)

    def process(self, audio_files_dir):
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                print(f"Processed file {file_path}")

    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        signal = self.bandpass.filter(signal)
        feature = self.extractor.extract(signal, self.loader.sample_rate)
        if self.normalizer:
            feature = self.normalizer.normalize(feature)
        feature, _ = self.selector.select(feature, self.extractor.n_fft, self.loader.sample_rate)
        features = self.segmenter.segment(feature, self.loader.sample_rate, self.extractor.hop_length)
        for f in features:
            self.saver.save_feature(f, file_path)

    def _is_padding_necessary(self, signal):
        if len(signal) < self._num_expected_samples:
            return True
        return False

    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal


if __name__ == "__main__":
    SAMPLE_RATE = 44100

    nFFT = 128
    WIN_LENGTH = 128
    HOP_LENGTH = 96
    spec_mode = "spectrogram"

    # norm_mode = "mean_std"

    MIN_FREQ = 500
    MAX_FREQ = 14000

    SEGMENT_SIZE = 0.1
    SEGMENT_HOP_SIZE = 0.025

    SPECTROGRAMS_SAVE_DIR = "../Dataset/NAU dataset/TASCAM/lab/Ponderosa/96 KHz/preprocessed_dataset/infested/"
    FILES_DIR = "../Dataset/NAU dataset/TASCAM/lab/Ponderosa/96 KHz/infested log longhorn/"

    # instantiate all objects
    loader = Loader(SAMPLE_RATE)
    padder = Padder()
    bandpass = BandpassFilter(MIN_FREQ, MAX_FREQ, SAMPLE_RATE, order=2)
    spectrogram_extractor = SpectrogramExtractor(nFFT, WIN_LENGTH, HOP_LENGTH, mode=spec_mode)
    # normalizer = Normalizer()
    selector = FrequencySelector(MIN_FREQ, MAX_FREQ)
    segmenter = SpectrogramSegmenter(SEGMENT_SIZE, SEGMENT_HOP_SIZE, SAMPLE_RATE, HOP_LENGTH)
    saver = Saver(SPECTROGRAMS_SAVE_DIR)

    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.bandpass = bandpass
    preprocessing_pipeline.extractor = spectrogram_extractor
    # preprocessing_pipeline.normalizer = normalizer
    preprocessing_pipeline.selector = selector
    preprocessing_pipeline.segmenter = segmenter
    preprocessing_pipeline.saver = saver

    preprocessing_pipeline.process(FILES_DIR)
