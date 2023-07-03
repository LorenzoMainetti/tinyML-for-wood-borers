import numpy as np
from random import uniform, randint
from scipy.io import wavfile
from Pipeline.data_augmentation import SignalAugmenter


class DataGenerator:
    """
    wood_sound: Wood sound time series
    insect_sound: Insect sound time series
    detection_label: Detection label: 0 = no insect, 1 = insect
    class_label: Insect species label
    sample_rate: Sampling rate of the generated data
    pulse_num_range: Range of number of pulses to insert
    inter_pulse_duration: Range of silence between pulses
    group_num_range: Range of number of groups to insert
    group_duration: Range of group duration
    inter_group_duration: Range of silence between groups
    """

    def __init__(
            self,
            wood_sound,
            insect_sound,
            detection_label,
            class_label=None,
            sample_rate=44100,
            pulse_num_range=(1, 10),
            inter_pulse_duration=(0.1, 0.6),
            group_num_range=(1, 3),
            group_duration=(0.8, 2.3),
            inter_group_duration=(0.8, 1.5)
    ):
        # set the wood and insect sound
        self.wood_sound = wood_sound
        self.insect_sound = insect_sound

        # set the detection label and the class label
        self.detection_label = detection_label
        self.class_label = class_label

        # set the parameters for the data generation (domain knowledge)
        self.sample_rate = sample_rate
        self.pulse_num_range = pulse_num_range
        self.inter_pulse_duration = inter_pulse_duration
        self.group_num_range = group_num_range
        self.group_duration = group_duration
        self.inter_group_duration = inter_group_duration

        # TODO check if wood sound and insect sound have the same sampling rate

    def set_wood_sound(self, wood_sound):
        self.wood_sound = wood_sound

    def set_insect_sound(self, insect_sound):
        self.insect_sound = insect_sound

    def generate_data(self, n_files, duration, save_path, save_num=0, mode='augment'):
        """
        Generate a dataset of desired length
        :param n_files: number of files to generate
        :param duration: duration of each file
        :param save_path: path where to save the generated files
        :param save_num: number to start the file naming
        :param mode: choose between 'random_noise', 'augment', 'standardize' or 'none'
        """
        n_samples = int(duration * self.sample_rate)

        for i in range(n_files):
            data = self.generate_background(n_samples)
            if self.detection_label == 1:
                data = self.insert_pulses(data.copy())

            if mode == 'random_noise':
                data = self.add_random_noise(data, noise_level=0.0001)
            elif mode == 'augment':
                data = self.randomly_augment_data(data, pulse=False)
            elif mode == 'standardize':
                data = (data - np.mean(data)) / np.std(data)

            assert (len(data) == n_samples)

            wavfile.write(
                filename=save_path + self.class_label + "_" + str(save_num + i) + ".wav",
                rate=self.sample_rate,
                data=data.astype(np.float32)
            )

    def generate_background(self, n_samples):
        """
        Generate a background time series of desired length
        :param n_samples: length of the time series
        :return: background time series
        """
        if len(self.wood_sound) == 0:
            raise ValueError("Wood sound is empty")

        # if the wood sound is longer than the required duration, take a random part of it
        if len(self.wood_sound) >= n_samples:
            # randomly select the starting point of the wood sound
            start = np.random.randint(0, len(self.wood_sound) - n_samples)
            wood_sound = self.wood_sound[start:start + n_samples]
            background = wood_sound
        else:
            wood_sound = self.wood_sound
            # generate a time series of desired length by repeating the original series
            background = self.repeat_time_series(wood_sound, n_samples)

        mean = np.mean(wood_sound)
        dim = len(wood_sound)

        # randomly modify the copied values adding or subtracting a random value in the range [-mean, +mean]
        background[dim:] = background[dim:] + np.random.uniform(-mean, mean, n_samples - dim)

        return background

    def insert_pulses(self, audio):
        """
        Randomly insert pulses in the background, respecting the pulse_num_range and inter_pulse_duration
        :param audio: background time series
        :return: background with pulses inserted
        """
        if len(self.insect_sound) >= len(audio):
            # if the insect sound has same length as the background, take the first n_samples
            start = 0
            if len(self.insect_sound) > len(audio):
                # if the insect sound is longer than the background, take a random part of it
                start = np.random.randint(0, len(self.insect_sound) - len(audio))

            # apply data augmentation that preserves the length
            insect_sound = self.randomly_augment_data(self.insect_sound, pulse=False)
            audio = insect_sound[start:start + len(audio)]

        else:
            num_pulses = np.random.randint(self.pulse_num_range[0], self.pulse_num_range[1])
            pulse_start = np.random.randint(0, len(audio) - len(self.insect_sound))

            for i in range(num_pulses):
                pulse = self.randomly_augment_data(self.insect_sound, pulse=True)
                if pulse_start + len(pulse) > len(audio):
                    continue
                else:
                    audio[pulse_start:pulse_start + len(pulse)] = audio[pulse_start:pulse_start + len(pulse)] + pulse

                if num_pulses > 1:
                    # compute the silence between the pulses
                    pulse_silence = int(
                        uniform(self.inter_pulse_duration[0], self.inter_pulse_duration[1]) * self.sample_rate)

                    # next pulse start must be after the silence and before the end of the audio
                    pulse_start = pulse_start + len(pulse) + pulse_silence
                    if pulse_start + len(self.insect_sound) > len(audio):
                        break

        return audio

    def insert_groups(self, audio):
        """
        Randomly insert groups of pulses in the background, respecting the group_num_range and inter_group_duration
        :param audio: background time series
        :return: background with groups of pulses inserted
        """
        num_groups = np.random.randint(self.group_num_range[0], self.group_num_range[1])
        group_start = np.random.randint(0, len(audio) - int(self.group_duration[1] * self.sample_rate))

        for i in range(num_groups):
            group = self.generate_group()

            if group_start + len(group) > len(audio):
                continue
            audio[group_start:group_start + len(group)] = audio[group_start:group_start + len(group)] + group

            if num_groups > 1:
                group_silence = int(
                    uniform(self.inter_group_duration[0], self.inter_group_duration[1]) * self.sample_rate)
                group_start = self.compute_next_start(group_start, group_silence, len(audio), len(group),
                                                      pulse_start_in=0)

        return audio

    def generate_group(self):
        """
        Generate a group of pulses respecting the pulse_num_range, inter_pulse_duration and group_duration
        :return: group of pulses
        """
        # randomly choose a group duration in the provided range
        group_duration = uniform(self.group_duration[0], self.group_duration[1])
        group_length = int(group_duration * self.sample_rate)

        # randomly choose the number of pulses in the group
        num_pulses = np.random.randint(self.pulse_num_range[0], self.pulse_num_range[1])

        # initialize the group with zeros
        group = np.zeros(group_length)
        # insert a pulse at the beginning and one at the end of the group
        group = self.insert_pulse_at_position(group, pulse_start=0)
        group = self.insert_pulse_at_position(group, pulse_start=group_length - len(self.insect_sound))

        # compute the remaining available length of the group
        remaining_length = group_length - 2 * (len(self.insect_sound) + self.inter_pulse_duration[0] * self.sample_rate)
        # compute the lower bound (for the range) for the next pulse start position
        pulse_start_in = len(self.insect_sound) + int(self.inter_pulse_duration[0] * self.sample_rate)
        # initialize the old start position to 0
        old_start = 0

        if num_pulses > 2:
            # insert the other pulses in the middle of the group randomly but respecting the inter pulse duration
            for i in range(num_pulses - 2):
                # randomly choose the inter pulse duration
                pulse_silence = int(
                    uniform(self.inter_pulse_duration[0], self.inter_pulse_duration[1]) * self.sample_rate)
                # compute the next pulse start position
                pulse_start = self.compute_next_start(
                    old_start,
                    pulse_silence,
                    remaining_length,
                    len(self.insect_sound),
                    pulse_start_in=pulse_start_in
                )
                # insert the pulse at the computed position
                group = self.insert_pulse_at_position(group, pulse_start)
                # update the old start position
                old_start = pulse_start

        return group

    def insert_pulse_at_position(self, audio, pulse_start):
        """
        Insert a pulse at the given position in the time series
        :param audio: time series
        :param pulse_start: pulse start position
        :return: time series with pulse inserted at the given position
        """
        pulse = self.randomly_augment_data(self.insect_sound, pulse=True)

        # if the pulse start exceeds the time series size, don't insert it
        if pulse_start + len(pulse) > len(audio):
            return audio
        else:
            audio[pulse_start:pulse_start + len(pulse)] = audio[pulse_start:pulse_start + len(pulse)] + pulse
            return audio

    @staticmethod
    def compute_next_start(old_start, silence, n_samples, length, pulse_start_in=0):
        """
        Compute the next pulse start position
        :param old_start: old start position
        :param silence: randomly selected inter_pulse_duration or inter_group_duration
        :param n_samples: number of samples in the time series
        :param length: length of the pulse or group of pulses
        :param pulse_start_in: lower bound for the next pulse start position
        :return: next pulse start position
        """
        pulse_start_fin = n_samples - length - silence

        pulse_start = np.random.randint(pulse_start_in, pulse_start_fin)
        if np.abs(pulse_start - old_start) < silence:
            pulse_start = old_start + silence

        return pulse_start

    @staticmethod
    def repeat_time_series(series, length):
        """
        Repeat a time series to obtain a time series of the desired length
        :param series: time series
        :param length: desired length
        :return: repeated time series
        """
        repeat_count = length // len(series) + 1
        repeated_series = np.tile(series, repeat_count)
        return repeated_series[:length]

    def randomly_augment_data(self, data, pulse=False):
        """
          Randomly augment a signal with one randomly selected technique
          :param data: data to augment
          :param pulse: control which augmentation to avoid
          :return: augmented data
          """
        if pulse:
            choices = ['noise_inj', 'volume_ctl', 'pitch_shift', 'time_shift', 'time_stretch',
                       'clipping', 'distorted_clipping', 'fading', 'time_inversion', 'none']
        else:
            choices = ['noise_inj', 'volume_ctl', 'pitch_shift', 'fading', 'none']

        # randomly sample a random number of aug techniques without replacement from the choices list
        num_augs = np.random.randint(1, len(choices))
        random_choice = np.random.choice(choices, num_augs, replace=False)

        # apply the randomly selected augmentation techniques
        for c in random_choice:
            data = self.apply_augmentation(data, c)

        return data

    def apply_augmentation(self, audio, random_choice):
        aug = SignalAugmenter(audio, self.sample_rate)

        if random_choice == 'noise_inj':
            return aug.noise_injection(uniform(0, 0.0004))
        elif random_choice == 'volume_ctl':
            return aug.volume_control(low=0.4, high=3)
        elif random_choice == 'pitch_shift':
            return aug.pitch_shifting(uniform(-10, -0.1))
        elif random_choice == 'time_shift':
            return aug.time_shifting(randint(-4000, 4000))
        elif random_choice == 'time_stretch':
            return aug.time_stretching(uniform(0.4, 2))
        elif random_choice == 'clipping':
            return aug.clipping(threshold=uniform(0.0012, 0.003))
        elif random_choice == 'distorted_clipping':
            return aug.distorted_clipping(percentile_threshold=randint(3, 10))
        elif random_choice == 'fading':
            return aug.fading(
                min_gain_in_db=uniform(-5.0, 0.0),
                max_gain_in_db=uniform(6.0, 12.0),
            )
        elif random_choice == 'time_inversion':
            return aug.time_inversion()
        else:
            return audio

    @staticmethod
    def add_random_noise(audio, noise_level=0.01):
        """
        Add random noise to a time series
        :param audio: time series
        :param noise_level: noise level
        :return: time series with random noise added
        """
        noise = np.random.normal(0, noise_level, len(audio))
        return audio + noise
