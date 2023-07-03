import os

import librosa
import numpy as np
import tensorflow as tf
import glob

from sklearn.model_selection import train_test_split

from Pipeline.data_augmentation import SpectrogramAugmenter
from Pipeline.preprocess import Loader, BandpassFilter, Normalizer, SpectrogramExtractor, FrequencySelector


class RawDatasetBuilder:
    """
    Builds a non tf dataset from a directory of audio files
    """

    def __init__(self, dataset_dir, class_dict, seed, sample_rate=44100, balanced=False):
        self.dataset_dir = dataset_dir
        self.class_dict = class_dict
        self.seed = seed
        self.balanced = balanced
        self.loader = Loader(sample_rate=sample_rate)

    def build(self, val_split=0.2, test_split=0.1):
        features, labels = self.get_features_and_labels()
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_split,
                                                            random_state=self.seed)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_split, random_state=self.seed)

        train_dataset = [X_train, y_train]
        val_dataset = [X_val, y_val]
        test_dataset = [X_test, y_test]

        return train_dataset, val_dataset, test_dataset

    def get_features_and_labels(self):
        if self.balanced:
            features, labels = self._balanced_extraction()
        else:
            features, labels = self._unbalanced_extraction()

        return features, labels

    def _unbalanced_extraction(self):
        features = []
        labels = []

        for path in glob.glob(f'{self.dataset_dir}', recursive=True):
            features.append(self.loader.load(path))
            for key in self.class_dict:
                if key in path:
                    labels.append(self.class_dict[key])
                    break

        return features, labels

    def _balanced_extraction(self):
        # retrieve the paths of each class
        folders_path = self.dataset_dir.split("*")[0]
        paths = []
        for key in self.class_dict.keys():
            paths.append(folders_path + key + '/')

        # find the lengths of each folder of each class
        lengths = []
        for path in paths:
            lengths.append(len(os.listdir(path)))

        # take the minimum of these lengths
        min_length = min(lengths)

        features = []
        labels = []
        keys = list(self.class_dict.keys())
        # randomly select min_length files from each folder and generate features and labels
        for i, path in enumerate(paths):
            files = os.listdir(path)
            files = np.random.choice(files, min_length, replace=False)
            for file in files:
                features.append(self.loader.load(path + file))
                labels.append(self.class_dict[keys[i]])

        return features, labels


class DatasetBuilder:
    def __init__(self, batch_size, seed, dataset_dir=None,
                 shuffle=True, augment=False, balanced=False, multi_task=False):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.augment = augment
        self.balanced = balanced
        self.multi_task = multi_task

    def build(self, val_split=0.2, test_split=0.2):
        features, labels = self._get_features_and_labels()

        if self.multi_task:
            # detection label is 0 if label is 0 (clean), 1 (infested) otherwise
            detection_labels = np.array(labels, copy=True)
            detection_labels[detection_labels != 0] = 1

            X_train, X_test, d_label_train, d_label_test, c_label_train, c_label_test = \
                train_test_split(features, detection_labels, labels, test_size=test_split, random_state=self.seed)
            X_train, X_val, d_label_train, d_label_val, c_label_train, c_label_val = \
                train_test_split(X_train, d_label_train, c_label_train, test_size=val_split, random_state=self.seed)

            y_train = {
                'detection': d_label_train,
                'classification': c_label_train
            }
            y_val = {
                'detection': d_label_val,
                'classification': c_label_val
            }
            y_test = {
                'detection': d_label_test,
                'classification': c_label_test
            }

        else:
            X_train, X_test, y_train, y_test = \
                train_test_split(features, labels, test_size=test_split, random_state=self.seed)
            X_train, X_val, y_train, y_val = \
                train_test_split(X_train, y_train, test_size=val_split, random_state=self.seed)

        train_dataset = self.build_dataset(X_train, y_train, train=True)
        val_dataset = self.build_dataset(X_val, y_val, train=False)

        # the test dataset is not converted to a tf.data.Dataset
        X_test = np.expand_dims(X_test, axis=-1).astype(np.float32)
        test_dataset = [X_test, y_test]

        return train_dataset, val_dataset, test_dataset

    def build_dataset(self, features, labels, train=False):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.map(self._add_axis)
        if self.shuffle:
            buffer_size = dataset.cardinality().numpy()
            dataset = dataset.shuffle(buffer_size=buffer_size, seed=self.seed)
        if self.augment and train:
            # dataset = dataset.map(self._py_wrapper)  TODO it degenerates the performance
            dataset = dataset.map(self._apply_image_augment)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def _get_features_and_labels(self):
        raise NotImplementedError

    def _apply_spec_augment(self, feature):
        # convert feature to numpy array
        feature = tf.squeeze(feature).numpy()

        # apply spec augment to the spectrogram
        aug = SpectrogramAugmenter(feature)
        feature = aug.rand_spec_augment(self.seed)

        # convert feature back to tensor
        feature = tf.convert_to_tensor(feature)
        feature = feature[..., tf.newaxis]
        return feature

    def _py_wrapper(self, feature, label):
        [feature, ] = tf.py_function(self._apply_spec_augment, inp=[feature], Tout=[tf.float32])
        return feature, label

    def _apply_image_augment(self, feature, label):
        # apply image augment to the spectrogram
        # 1. random mirroring
        feature = tf.image.random_flip_left_right(feature, seed=self.seed)
        # 2. random flipping
        feature = tf.image.random_flip_up_down(feature, seed=self.seed)
        return feature, label

    @staticmethod
    def _add_axis(feature, label):
        # Add a `channels` dimension, so that the spectrogram can be used
        # as image-like input data with convolution layers which expect
        # shape (`batch_size`, `height`, `width`, `channels`).
        feature = feature[..., tf.newaxis]
        return feature, label


class DatasetBuilderRaw(DatasetBuilder):
    """
    DatasetBuilder from a directory of .wav files, where each file contains a single audio recording
    """

    def __init__(self, dataset_dir, batch_size, seed, class_dict, min_freq=200, max_freq=20000, sample_rate=44100,
                 shuffle=True, augment=False, balanced=False, multi_task=False, normalize=True):
        super().__init__(batch_size, seed, dataset_dir, shuffle, augment, balanced, multi_task)
        self.raw_dataset_builder = RawDatasetBuilder(
            dataset_dir=dataset_dir,
            class_dict=class_dict,
            seed=seed,
            sample_rate=sample_rate,
            balanced=balanced,
        )
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.sample_rate = sample_rate
        self.normalize = normalize

    def _get_features_and_labels(self):
        features, labels = self.raw_dataset_builder.get_features_and_labels()

        # apply bandpass filter to each feature
        bandpass = BandpassFilter(self.min_freq, self.max_freq, self.sample_rate, order=2)
        normalizer = Normalizer(mode='mean_std')
        features = [bandpass.filter(feature) for feature in features]
        if self.normalize:
            features = [normalizer.normalize(feature) for feature in features]

        features = np.array(features).astype(np.float32)
        labels = np.array(labels)

        return features, labels


class DatasetBuilderSpec(DatasetBuilder):
    """
    DatasetBuilder from a directory of .npy files, where each file contains a spectrogram
    """

    def __init__(self, dataset_dir, batch_size, seed, shuffle=True, augment=False, balanced=False, log=True):
        super().__init__(batch_size, seed, dataset_dir, shuffle, augment, balanced)
        self.log = log

    def _get_features_and_labels(self):
        features = []
        labels = []
        for path in glob.glob(f'{self.dataset_dir}', recursive=True):
            spectrogram = np.load(path)
            if not self.log:
                spectrogram = librosa.db_to_amplitude(spectrogram)
            features.append(spectrogram)
            if "clean" in str(path):
                labels.append(0)
            else:
                labels.append(1)

        features = np.array(features)
        labels = np.array(labels)

        if self.balanced:
            features, labels = self._balance(features, labels)

        return features, labels

    @staticmethod
    def _balance(features, labels):
        # Balance the dataset
        clean = features[labels == 0]
        infested = features[labels == 1]
        clean = clean[np.random.choice(clean.shape[0], len(infested), replace=False)]
        features = np.concatenate((clean, infested))
        labels = np.concatenate((np.zeros(len(clean)), np.ones(len(infested))))
        return features, labels


class DatasetBuilderPipeline(DatasetBuilder):
    """
    DatasetBuilder from a directory of raw .wav files (spectrogram extraction is done on the fly)
    """

    def __init__(self, dataset_dir, batch_size, class_dict, seed,
                 sample_rate=44100,
                 nfft=128, win_length=128, hop_length=96,
                 min_freq=500, max_freq=14000,
                 shuffle=True, augment=False, balanced=False, log=True, multi_task=False):
        super().__init__(batch_size, seed, dataset_dir, shuffle, augment, balanced, multi_task)
        self.sample_rate = sample_rate
        self.nfft = nfft
        self.win_length = win_length
        self.hop_length = hop_length
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.log = log

        self.raw_dataset_builder = RawDatasetBuilder(
            dataset_dir=dataset_dir,
            class_dict=class_dict,
            seed=seed,
            sample_rate=sample_rate,
            balanced=balanced,
        )

    def _get_features_and_labels(self):
        raw_features, labels = self.raw_dataset_builder.get_features_and_labels()
        features = []

        for f in raw_features:
            features.append(self._preprocess(f))

        features = np.array(features)
        labels = np.array(labels)

        return features, labels

    def _preprocess(self, signal):
        bandpass = BandpassFilter(self.min_freq, self.max_freq, self.sample_rate, order=2)
        normalizer = Normalizer(mode='mean_std')
        spectrogram_extractor = SpectrogramExtractor(self.nfft, self.win_length, self.hop_length, mode="spectrogram")
        selector = FrequencySelector(self.min_freq, self.max_freq)

        signal = bandpass.filter(signal)
        spectrogram = spectrogram_extractor.extract(signal, self.sample_rate, log=self.log).astype(np.float32)
        spectrogram = normalizer.normalize(spectrogram)
        spectrogram, _ = selector.select(spectrogram, self.nfft, self.sample_rate)

        return spectrogram
