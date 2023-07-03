import numpy as np


def hopping_window(signal, frame_length, hop_length):
    splits = []
    for i in range(0, len(signal), hop_length):
        if len(signal[i:i + frame_length]) < frame_length:
            break
        splits.append(signal[i:i + frame_length])
    return np.asarray(splits)
