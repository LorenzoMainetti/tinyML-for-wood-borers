import random
import audiomentations
import librosa
import numpy as np
from pydub import AudioSegment
import tensorflow as tf
from tensorflow_addons.image import sparse_image_warp


class SignalAugmenter:
    def __init__(self, audio, sr):
        self.audio = audio
        self.sr = sr

    # Basic signal augmentation
    def noise_injection(self, noise=0.009):
        wav_n = self.audio + noise * np.random.normal(0, 1, len(self.audio))
        return wav_n

    def volume_control(self, low=1.5, high=3):
        wav_v = self.audio * np.random.uniform(low=low, high=high)
        return wav_v

    def time_shifting(self, shift):
        wav_roll = np.roll(self.audio, shift)
        return wav_roll

    def time_stretching(self, factor, noise=0.00005):
        wav_time_stretch = librosa.effects.time_stretch(y=self.audio, rate=factor)
        wav_time_stretch = wav_time_stretch + noise * np.random.normal(0, 1, len(wav_time_stretch))
        return wav_time_stretch

    def pitch_shifting(self, n_steps=-5.0):
        wav_pitch_sf = librosa.effects.pitch_shift(y=self.audio, sr=self.sr, n_steps=n_steps)
        return wav_pitch_sf

    # Advanced signal augmentation
    def wow_resampling(self, alpha=3, beta=2):
        # TODO: check if this is correct
        wav_res = self.audio + alpha * (np.sin(2 * np.pi * beta * self.audio) / 2 * np.pi * beta)
        return wav_res

    def amplitude_attenuation(self, distance, alpha):
        # TODO: check if this is correct
        attenuation = alpha / distance
        return self.audio * attenuation

    def harmonic_distortion(self):
        # TODO: check if this is correct
        return np.sin(np.sin(np.sin(np.sin(np.sin(2 * np.pi * self.audio)))))

    def clipping(self, threshold=1):
        wav_clipped = np.clip(self.audio, -threshold, threshold)
        return wav_clipped

    def distorted_clipping(self, percentile_threshold=40):
        clipper = audiomentations.ClippingDistortion(
            min_percentile_threshold=0, max_percentile_threshold=percentile_threshold, p=1)
        clipper.randomize_parameters(self.audio, self.sr)
        return clipper.apply(self.audio, self.sr)

    def fading(self, min_gain_in_db=-24.0, max_gain_in_db=6.0, min_duration=0.2, max_duration=6.0):
        gain_transition = audiomentations.GainTransition(
            min_gain_in_db=min_gain_in_db,
            max_gain_in_db=max_gain_in_db,
            min_duration=min_duration,
            max_duration=max_duration,
            p=1
        )
        gain_transition.randomize_parameters(self.audio, self.sr)
        return gain_transition.apply(self.audio, self.sr)

    def dynamic_range_compression(self, min_threshold_db=-24, max_threshold_db=-2):
        # TODO: issue with Cylimiter
        dynamic_range_compression = audiomentations.Limiter(
            min_threshold_db=min_threshold_db,
            max_threshold_db=max_threshold_db,
            p=1)
        return dynamic_range_compression.apply(self.audio, self.sr)

    def time_inversion(self):
        # widely used in the visual domain. This can be relevant in the context of audio classification
        time_inversion = audiomentations.Reverse(p=1)
        return time_inversion.apply(self.audio, self.sr)


class SpectrogramAugmenter:
    def __init__(self, spectrogram):
        self.spectrogram = spectrogram.astype(np.float32)

    def frequency_masking(self, min_mask_fraction=0.03, max_mask_fraction=0.15, fill_mode='mean'):
        freq_mask = audiomentations.SpecFrequencyMask(
            min_mask_fraction=min_mask_fraction,
            max_mask_fraction=max_mask_fraction,
            fill_mode=fill_mode,
            p=1)
        freq_mask.randomize_parameters(self.spectrogram)
        return freq_mask.apply(self.spectrogram)

    def time_masking(self, min_mask_fraction=0.03, max_mask_fraction=0.15, fill_mode='mean'):
        time_mask = SpecTimeMask(
            min_mask_fraction=min_mask_fraction,
            max_mask_fraction=max_mask_fraction,
            fill_mode=fill_mode,
        )
        time_mask.randomize_parameters(self.spectrogram)
        return time_mask.apply(self.spectrogram)

    def rand_freq_shifting(self):
        shift = np.random.randint(0, self.spectrogram.shape[0])
        s1 = self.spectrogram[shift:, :]
        s2 = self.spectrogram[:shift, :]
        return np.concatenate((s1, s2), axis=0)

    def rand_time_shifting(self):
        shift = np.random.randint(0, self.spectrogram.shape[1])
        s1 = self.spectrogram[:, shift:]
        s2 = self.spectrogram[:, :shift]
        return np.concatenate((s1, s2), axis=1)

    def time_warping(self, warp=10):
        # Reshape to [Batch_size, time, freq, 1] for sparse_image_warp func.
        spectrogram = np.reshape(self.spectrogram, (-1, self.spectrogram.shape[0], self.spectrogram.shape[1], 1))

        v, tau = spectrogram.shape[1], spectrogram.shape[2]
        assert(tau > warp)

        horiz_line_thru_ctr = spectrogram[0][v // 2]

        # Random point along the horizontal/time axis
        random_pt = horiz_line_thru_ctr[random.randrange(warp, tau - warp)]
        # Distance from the random point to the left and right side of the spectrogram
        w = np.random.uniform((-warp), warp)

        # Source Points
        src_points = [[[v // 2, random_pt[0]]]]
        # Destination Points
        dest_points = [[[v // 2, random_pt[0] + w]]]

        spectrogram = tf.convert_to_tensor(spectrogram)
        src_points = tf.convert_to_tensor(src_points)
        dest_points = tf.convert_to_tensor(dest_points)

        spectrogram, _ = sparse_image_warp(spectrogram, src_points, dest_points, num_boundary_points=2)
        return tf.squeeze(spectrogram).numpy()

    def rand_spec_augment(self, seed):
        np.random.seed(seed)

        # randomize augmentation
        if np.random.rand() < 0.5:
            self.spectrogram = self.frequency_masking()
        if np.random.rand() < 0.5:
            self.spectrogram = self.time_masking()
        if np.random.rand() < 0.5:
            self.spectrogram = self.rand_freq_shifting()
        if np.random.rand() < 0.5:
            self.spectrogram = self.rand_time_shifting()
        # if np.random.rand() < 0.5:
            # self.spectrogram = self.time_warping()
        return self.spectrogram


class MixupAugmenter:
    """
    Create a new audio sample by overlaying two audio samples
    """

    def __init__(self, path1, path2):
        audio1 = AudioSegment.from_wav(path1)
        audio2 = AudioSegment.from_wav(path2)
        # if the length is different, pad the shortest file
        if len(audio1) > len(audio2):
            audio2 = audio2 + AudioSegment.silent(duration=len(audio1) - len(audio2))
        elif len(audio1) < len(audio2):
            audio1 = audio1 + AudioSegment.silent(duration=len(audio2) - len(audio1))

        self.audio1 = audio1
        self.audio2 = audio2

    def mixup(self, position=0, mixed_gain=0):
        audio2 = self.audio2.apply_gain(1 - mixed_gain)
        mixed_audio = self.audio1.overlay(audio2, position=position, gain_during_overlay=mixed_gain)
        return self.audiosegment_to_ndarray(mixed_audio)

    @staticmethod
    def audiosegment_to_ndarray(audiosegment):
        channel_sounds = audiosegment.split_to_mono()
        samples = [s.get_array_of_samples() for s in channel_sounds]

        fp_arr = np.array(samples).T.astype(np.float32)
        fp_arr /= np.iinfo(samples[0].typecode).max
        fp_arr = fp_arr.reshape(-1)

        return fp_arr


class SpecTimeMask:
    def __init__(
            self,
            min_mask_fraction: float = 0.03,
            max_mask_fraction: float = 0.25,
            fill_mode: str = "constant",
            fill_constant: float = 0.0,
    ):
        self.parameters = {}
        self.min_mask_fraction = min_mask_fraction
        self.max_mask_fraction = max_mask_fraction
        assert fill_mode in ("mean", "constant")
        self.fill_mode = fill_mode
        self.fill_constant = fill_constant

    def randomize_parameters(self, magnitude_spectrogram):
        num_time_bins = magnitude_spectrogram.shape[1]
        min_times_to_mask = int(round(self.min_mask_fraction * num_time_bins))
        max_times_to_mask = int(round(self.max_mask_fraction * num_time_bins))

        num_times_to_mask = random.randint(min_times_to_mask, max_times_to_mask)
        self.parameters["start_time_index"] = random.randint(0, num_time_bins - num_times_to_mask)
        self.parameters["end_time_index"] = (self.parameters["start_time_index"] + num_times_to_mask)

    def apply(self, magnitude_spectrogram):
        if self.fill_mode == "mean":
            fill_value = np.mean(
                magnitude_spectrogram[:, self.parameters["start_time_index"]:self.parameters["end_time_index"]]
            )
        else:
            # self.fill_mode == "constant"
            fill_value = self.fill_constant

        magnitude_spectrogram = magnitude_spectrogram.copy()
        magnitude_spectrogram[:, self.parameters["start_time_index"]: self.parameters["end_time_index"]] = fill_value

        return magnitude_spectrogram
