import librosa
import numpy as np
import pandas as pd
import scipy
from pyod.models import ecod

from Pipeline.windowing import hopping_window


class Detector:
    def set_params(self, **params):
        pass

    def detect(self, data):
        pass


class ECODetector(Detector):
    def __init__(self, train_signal):
        self.train_signal = train_signal
        self.contamination = None
        self.detector = None

    def set_params(self, contamination):
        self.detector = ecod.ECOD(contamination=contamination)
        self.detector.fit(self.train_signal.reshape(-1, 1))

    def detect(self, data):
        predictions = []

        for sample in data:
            max_value = self.detector.decision_function(sample.reshape(-1, 1)).max()
            if max_value > self.detector.threshold_:
                predictions.append(True)
            else:
                predictions.append(False)

        return predictions


class USTEnergyDetector(Detector):
    def __init__(self, reference_noise, version='mean_noise_thresh'):
        self.reference_noise = reference_noise
        self.noise_mean_e = None
        self.noise_std_e = None

        self.version = version
        self.frame_length = None
        self.hop_length = None
        self.factor = None

    def set_params(self, frame_length, hop_length, factor):
        if hop_length > frame_length:
            raise ValueError('hop_length must be smaller than frame_length')

        self.frame_length = frame_length
        self.hop_length = hop_length
        self.factor = factor

        self.noise_mean_e, self.noise_std_e = self._compute_noise_threshold(self.reference_noise)

    def _short_time_energy(self, signal):
        frames = hopping_window(signal, self.frame_length, self.hop_length)

        return np.array([
            sum(abs(frames[i] ** 2))
            for i in range(len(frames))
        ])

    def _compute_noise_threshold(self, noise):
        noise_short_term_e = self._short_time_energy(noise)
        noise_mean_e = np.mean(noise_short_term_e)
        noise_std_e = np.std(noise_short_term_e)

        return noise_mean_e, noise_std_e

    def detect(self, data):
        predictions = []
        for signal in data:
            signal_short_term_e = self._short_time_energy(signal)

            if self.version == 'mean_noise_thresh':
                if max(signal_short_term_e) > self.factor * self.noise_mean_e:
                    predictions.append(True)
                else:
                    predictions.append(False)
            elif self.version == 'noise_distance':
                distance = []
                for i in range(len(signal_short_term_e)):
                    distance.append(np.abs(signal_short_term_e[i] - self.noise_mean_e) / self.noise_std_e)
                distance = np.array(distance)
                outliers = np.where(distance > self.factor)[0]

                if len(outliers) > 0:
                    predictions.append(True)
                else:
                    predictions.append(False)

        return predictions


class HilbertDetector(Detector):
    def __init__(self):
        self.factor = None

    def set_params(self, factor):
        self.factor = factor

    def detect(self, data):
        predictions = []
        for signal in data:
            hilbert = scipy.signal.hilbert(signal)
            envelope = np.real(np.sqrt(hilbert * np.conj(hilbert)))
            # threshold is calculated as thresh = 10*theta1 where theta1 is the mean
            # of the 80% of the lowest values of the envelope.
            n = int(0.8 * len(envelope))
            sorted_envelop = np.sort(envelope)
            threshold = self.factor * np.mean(sorted_envelop[:n])

            if np.max(envelope) > threshold:
                predictions.append(True)
            else:
                predictions.append(False)

        return predictions


class ECDFOutlierCountDetector(Detector):
    def __init__(self, reference_noise):
        self.reference_noise = reference_noise
        self.threshold = None
        self.outliers_count = None

    def set_params(self, pfa, outliers_count):
        self.threshold = self._compute_threshold(pfa)
        self.outliers_count = outliers_count

    def _compute_threshold(self, pfa):
        noise_squared = np.square(self.reference_noise)
        xcdf = np.sort(noise_squared)
        threshold = np.quantile(xcdf, 1 - pfa)
        return threshold

    def detect(self, data):
        predictions = []
        for signal in data:
            squared_sample = np.square(signal)
            num_outliers = np.count_nonzero(squared_sample > self.threshold)
            if num_outliers > self.outliers_count:
                predictions.append(True)
            else:
                predictions.append(False)

        return predictions


class MovingStdDetector(Detector):
    def __init__(self, version='adaptive'):
        self.version = version
        self.window_size = None
        self.grace_period = None
        self.factor = None
        self.mode = None

    def set_params(self, window_size, grace_period, factor, mode='sma'):
        if grace_period >= window_size:
            raise ValueError('grace_period must be smaller than window_size')

        self.window_size = window_size
        self.grace_period = grace_period
        self.factor = factor
        self.mode = mode

    def detect(self, data):
        predictions = []
        for signal in data:
            ts_df = pd.DataFrame({'data': signal})
            if self.mode == 'sma':
                std = ts_df['data'].rolling(self.window_size, min_periods=self.grace_period).std()
            elif self.mode == 'ewma':
                std = ts_df['data'].ewm(span=self.window_size, min_periods=self.grace_period).std()
            else:
                raise ValueError('This type of moving average is not supported.')

            if self.version == 'adaptive':
                upper_bound = self.factor * std
                lower_bound = -self.factor * std
                is_outlier = (ts_df['data'] > upper_bound) | (ts_df['data'] < lower_bound)
                outliers = ts_df[is_outlier]

                if len(outliers) > 0:
                    predictions.append(True)
                else:
                    predictions.append(False)

            elif self.version == 'fixed':
                if std.max() > self.factor:
                    predictions.append(True)
                else:
                    predictions.append(False)

        return predictions


class STALTADetector(Detector):
    def __init__(self):
        self.sta_win = None
        self.lta_win = None
        self.threshold = None

    def set_params(self, sta_win, lta_win, threshold):
        if sta_win > lta_win:
            raise ValueError('STA window size must be smaller than LTA window size.')

        self.sta_win = sta_win
        self.lta_win = lta_win
        self.threshold = threshold

    def detect(self, data):
        predictions = []
        for signal in data:
            if len(signal) % (self.sta_win + self.lta_win) != 0:
                signal = signal[:-(len(signal) % (self.sta_win + self.lta_win))]

            STA = np.array([
                sum(abs(signal[i:i + self.sta_win] ** 2))
                for i in range(0, len(signal), self.sta_win + self.lta_win)
            ])
            LTA = np.array([
                sum(abs(signal[i:i + self.lta_win] ** 2))
                for i in range(self.sta_win, len(signal), self.sta_win + self.lta_win)
            ])

            R = np.divide(STA, LTA, out=np.zeros_like(STA), where=LTA != 0)

            if max(R) > self.threshold:
                predictions.append(True)
            else:
                predictions.append(False)

        return predictions


class SNRDetector(Detector):
    def __init__(self, reference_noise):
        self.reference_noise = reference_noise
        self.noise_rms = None
        self.frame_length = None
        self.hop_length = None
        self.snr = None

    def set_params(self, frame_length, hop_length, snr):
        if hop_length > frame_length:
            raise ValueError('hop_length must be smaller than frame_length.')

        noise_rms = librosa.feature.rms(
            y=self.reference_noise,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        self.noise_rms = noise_rms[:-1]

        self.frame_length = frame_length
        self.hop_length = hop_length
        self.snr = snr

    def detect(self, data):
        predictions = []
        for signal in data:
            signal_rms = librosa.feature.rms(
                y=signal,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )[0]
            signal_rms = signal_rms[:-1]

            noise_rms = self.noise_rms
            if len(signal_rms) < len(self.noise_rms):
                noise_rms = self.noise_rms[:len(signal_rms)]

            if len(signal_rms) > len(self.noise_rms):
                signal_rms = signal_rms[:len(self.noise_rms)]

            # SNR as squared ratio between the rms of the signal and the rms of the noise
            ratio = np.divide(signal_rms, noise_rms, out=np.zeros_like(signal_rms), where=noise_rms != 0)
            SNR = 20 * np.log10(np.square(ratio))

            if max(SNR) > self.snr:
                predictions.append(True)
            else:
                predictions.append(False)

        return predictions
