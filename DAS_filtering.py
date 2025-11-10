import numpy as np
from scipy.signal import butter, lfilter

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# filtering
def define_butterworth_highpass(cutoff, Fs):
    nyquist = Fs / 2
    order = 3
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', output='ba', analog=False)
    sos = butter(order, normal_cutoff, btype='high', output='sos', analog=False)
    return b, a, sos

def filtering(phase, phase_filtered_total, initial_filter_conditions, Fs, filter_cutoff_frequency):
    b, a, sos = define_butterworth_highpass(cutoff=filter_cutoff_frequency, Fs=Fs)

    if initial_filter_conditions is None:
        filtered_data = lfilter(b, a, phase, axis=0)
        phase_filtered_total = filtered_data
        initial_filter_conditions = True
    else:
        filtered_data = lfilter(b, a, phase, axis=0)
        phase_filtered_total = np.concatenate((phase_filtered_total, filtered_data), axis=0)

    return phase_filtered_total

def define_butterworth_highpass2(cutoff, Fs):
    Fs=10000
    nyquist = Fs / 2
    order = 3
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', output='ba', analog=False)
    sos = butter(order, normal_cutoff, btype='high', output='sos', analog=False)
    return b, a, sos



def filtering2(phase, phase_filtered_total, initial_filter_conditions, Fs, filter_cutoff_frequency):
    Fs=100000
    b, a, sos = define_butterworth_highpass2(cutoff=filter_cutoff_frequency, Fs=10000)

    if initial_filter_conditions is None:
        filtered_data = lfilter(b, a, phase, axis=0)
        phase_filtered_total = filtered_data
        initial_filter_conditions = True
    else:
        filtered_data = lfilter(b, a, phase, axis=0)
        phase_filtered_total = np.concatenate((phase_filtered_total, filtered_data), axis=0)

    return phase_filtered_total
