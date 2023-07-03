from scipy.signal import butter, sosfiltfilt


def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    sos = butter(order, normal_cutoff, btype='low', output='sos')
    y = sosfiltfilt(sos, data)
    return y


def butter_highpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    sos = butter(order, normal_cutoff, btype='high', output='sos')
    y = sosfiltfilt(sos, data)
    return y


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    y = sosfiltfilt(sos, data)
    return y
