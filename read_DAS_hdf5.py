import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
from DAS_filtering import moving_average,define_butterworth_highpass,filtering
from numpy.fft import fft, ifft
import h5py
import numpy as np
from scipy.signal import butter, filtfilt, welch, spectrogram, decimate

def get_locations(spatial_sampling=2.45):
    # ventouse is air valve
    # vidange is washout
    locations = {
        'Chamber1': int((154 + 120) / spatial_sampling),  # (position + offset)/spatial_sampling
        'Ventouse1': int((154 + 120 + 202) / spatial_sampling),
        'Vidange1': int((154 + 120 + 202 + 115) / spatial_sampling),

        'Chamber2': int((950 + 120) / spatial_sampling),

        'Chamber3': int((1054 + 120) / spatial_sampling),

        'Chamber4': int((1837 + 120) / spatial_sampling),
        'Vidange4': int((1837 + 120 + 309) / spatial_sampling),

        'Chamber5': int((2448 + 120) / spatial_sampling),

        'Chamber6': int((2521 + 120) / spatial_sampling),
        'Vidange6': int((2521 + 120 + 255) / spatial_sampling),

        'Chamber6a': int((2773 + 120) / spatial_sampling),
        'Ventouse6a': int((2521 + 120 + 255 + 270) / spatial_sampling),
        'Vidange6a': int((2521 + 120 + 255 + 270 + 184) / spatial_sampling),

        'Chamber7': int((3560 + 120) / spatial_sampling),
        'Ventouse7': int((3560 + 120 + 269) / spatial_sampling),
        'Vidange7': int((3560 + 120 + 269 + 60) / spatial_sampling),

        'Chamber8': int((4188 + 120) / spatial_sampling),

        # values for C9-C10 link were adjusted after Igors trip to Morocco in 10/09 - 14/09
        'Chamber9': int((4223 + 120) / spatial_sampling),  # chamber 9 was correct

        # location of detected leak on 11/09/2024; real distance measured by Mark, Georgia and Igor from Airvalve
        'Detected_leak_11_09_2024': int((4223 + 120 + 248 - (4 * spatial_sampling) - 8) / spatial_sampling),
        # previous value: 4223+120+248

        'Ventouse9': int((4223 + 120 + 248 - (4 * spatial_sampling)) / spatial_sampling),
        # previous value: 4223+120+248
        'Vidange9': int((4223 + 120 + 248 + 250) / spatial_sampling),
        'dig1_13_09_2024': int((4861 + 120 - (4 * spatial_sampling) - 50) / spatial_sampling),
        'dig2_13_09_2024': int((4861 + 120 - (4 * spatial_sampling) - 40) / spatial_sampling),

        # aux tapping location before chamber 10; real distance measured by Mark, Georgia and Igor
        '80m_to_Chamber10': int((1993)),
        '70m_to_Chamber10': int((1999)),
        '65m_to_Chamber10': int((2001)),
        '60m_to_Chamber10': int((2003)),
        '50m_to_Chamber10': int((2007)),
        '40m_to_Chamber10': int((2011)),
        '30m_to_Chamber10': int((2014)),
        '10m_to_Chamber10': int((2025)),

        'Chamber10': int((4861 + 120 - (4 * spatial_sampling)) / spatial_sampling),  # previous value: 4861+120

        'Chamber11': int((5332 + 120) / spatial_sampling),

        'Chamber12': int((5799 + 120) / spatial_sampling),
        'Vidange12': int((5799 + 120 + 392) / spatial_sampling),

        'Chamber13': int((6453 + 120) / spatial_sampling),
        'Vidange13': int((6453 + 120 + 52) / spatial_sampling),

        'Chamber14': int((7364 + 120) / spatial_sampling),

        'Chamber15': int((7963 + 120) / spatial_sampling),
        'Vidange15': int((7963 + 120 + 173) / spatial_sampling),
        'Ventouse15': int((7963 + 120 + 173 + 926) / spatial_sampling),

        'Chamber16': int((9100 + 120) / spatial_sampling),
    }

    return locations


def load_single_DAS_file(file_name):
    hf = h5py.File(file_name, 'r')
    n1 = hf.get('DAS')
    n2 = np.array(n1)
    n2 = n2 * (np.pi / 2 ** 15)
    #print(f'[HDF5 Processing] Integrate')
    n22 = np.cumsum(n2,axis=0)
    return n22

def list_hdf5_files_in_dir (file_path):
    #file_path = pathlib.Path("C:\Kate captures/2024-04-18/100 lpm 60s 2")
    list(file_path.iterdir())
    file_count = 0
    for item in file_path.iterdir():
        if item.is_file():
            if file_count == 0:
                file_names = [item]
            else:
                file_names.append(item)
            file_count = file_count + 1
    #print('[HDF5 Processing] Number of Files', file_count)
    return file_names

def load_multi_DAS_file(file_names, number_of_files, channels, Fs, ):
    count = 0
    data_DAS_filtered = np.zeros(((number_of_files[1]-number_of_files[0]) * Fs*10, channels[1]-channels[0]))

    for i in range(number_of_files[0],number_of_files[1],1):
        file_name = file_names[i]
        initial_filter_conditions = None
        downsampling_factor = 1
        sys.stdout.flush()
        print('[HDF5 Processing] Current Filename', file_name, flush=True)
        hf = h5py.File(file_name, 'r')
        n1 = hf.get('DAS')[:,channels[0]:channels[1]]
        n2 = np.array(n1)
        n2 = n2 * (np.pi / 2 ** 15)
        #print(f'[HDF5 Processing] Integrate')
        n22 = np.cumsum(n2, axis=0)
        #Fs = 20000
        filter_cutoff_frequency = 5
        phase_filtered_total = np.array([])
        initial_filter_conditions = None

        phase_filtered_total = filtering(n22, phase_filtered_total, initial_filter_conditions, Fs,
                                         filter_cutoff_frequency)

        data_DAS_filtered[count:count + Fs*10, :] = phase_filtered_total

        count += Fs*10
        if i % 100 == 0 :
            print(i)
    return data_DAS_filtered

def pred_FCN(model, DAS_fitlered_data, channels ):
    predic = np.zeros((channels[1]-channels[0],3))
    for channel in range(predic.shape[0]):
        DAS_10_second_in_Frequency = fft(DAS_fitlered_data[:, channel])
        DAS_10_second_in_Frequency = np.abs(DAS_10_second_in_Frequency[40:10002])
        DAS_10_second_in_Frequency = DAS_10_second_in_Frequency / DAS_10_second_in_Frequency.sum()
        sd = np.expand_dims(DAS_10_second_in_Frequency, axis=0)
        sss = np.concatenate((sd, sd), axis=0)
        ssss = np.expand_dims(sss, axis=1)
        MAtestsss = moving_average(ssss[0, 0, 0:3000], n=100)
        MAtestsss = np.expand_dims(MAtestsss, axis=0)
        MAtestsss = MAtestsss / MAtestsss.sum()
        MAtestsss = np.concatenate((MAtestsss, MAtestsss), axis=0)
        MAtestsss = np.expand_dims(MAtestsss, axis=1)
        pred = model.predict(MAtestsss)
        # print(model.predict(MAtestsss[:,:,:]))
        predic[channel, :] = pred[0]
        if channel % 1000 == 0:
            print(channel)
    return predic

def generate_training_set(leak_time_period, leak_channels_period,channels_search_period, overlap_rate,DAS_fitlered_data):
    number_of_training_samples = int((DAS_fitlered_data.shape[0]/overlap_rate)*(channels_search_period[1]-channels_search_period[0]))
    training_data = np.zeros((number_of_training_samples, 1, 7901))
    training_label = np.zeros(number_of_training_samples)
    count = 0

    for i in range(channels_search_period[0], channels_search_period[1], 1):

        for j in range(0, DAS_fitlered_data.shape[0], overlap_rate):

            sample = DAS_fitlered_data[j:j + 200000, i]
            DAS_10_second_in_Frequency = fft(sample)
            DAS_10_second_in_Frequency = np.abs(DAS_10_second_in_Frequency[40:10002])
            DAS_10_second_in_Frequency = DAS_10_second_in_Frequency / DAS_10_second_in_Frequency.sum()
            sd = np.expand_dims(DAS_10_second_in_Frequency, axis=0)
            sss = np.concatenate((sd, sd), axis=0)
            ssss = np.expand_dims(sss, axis=1)
            MAtestsss = moving_average(ssss[0, 0, 0:8000], n=100)
            MAtestsss = np.expand_dims(MAtestsss, axis=0)
            MAtestsss = MAtestsss / MAtestsss.sum()
            MAtestsss = np.concatenate((MAtestsss, MAtestsss), axis=0)
            MAtestsss = np.expand_dims(MAtestsss, axis=1)
            # pred = model.predict(MAtestsss)
            training_data[count, 0, :] = MAtestsss[0, 0, :]

            if i > leak_channels_period[0] and i < leak_channels_period[1] and j > leak_time_period[0] and j < leak_time_period[1]:
                training_label[count] = 1
                count += 1
            else:
                training_label[count] = 0
                count += 1

    return training_data, training_label


def fft_signal (DAS_fitlered_data, channels, time, fs):
    Time_intervals = DAS_fitlered_data.shape[0]/fs
    fft_array_real = np.zeros((int(Time_intervals), channels[1]-channels[0], 1000))
    ts = 1.0 / fs
    t = np.arange(0, 1, ts)
    count_ch = 0
    channels_range = DAS_fitlered_data.shape[1]
    for channel_idx in range(0, channels_range,1):
        count_t = 0
        for time_ids in range(0, DAS_fitlered_data.shape[0], time[1]):
            x = DAS_fitlered_data[time_ids:time_ids+time[1], channel_idx]
            X = (fft(x))
            N = len(X)
            n = np.arange(N)
            T = N / fs
            freq = n / T
            fft_array_real[count_t, count_ch] = X.real[0:1000]
            count_t+=1
        count_ch+=1
        #print("channel",count_ch)
    return fft_array_real, freq, t

def signal_Energy(fft_DAS_signals,channels,frequency_range,time_interval):
    Energy_array = np.zeros((fft_DAS_signals.shape[0],int(channels[1]-channels[0]), 1))
    for time_ids in range(0, fft_DAS_signals.shape[0], time_interval):
        for channel_idx in range(channels[0], channels[1], 1):
            Energy_array[time_ids, channel_idx, 0] = fft_DAS_signals[time_ids:time_ids+time_interval, channel_idx, frequency_range[1]- frequency_range[0] ].sum()
    return Energy_array


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def training_data_generation(fft_DAS_signals, time_big_leak, time_small_leak,channels_leak, frequency_range,channel_width,time_width  ):
    training_data = np.zeros((fft_DAS_signals.shape[0] * (fft_DAS_signals.shape[1] - 70), 10, 10, 1400))
    training_label = np.zeros(fft_DAS_signals.shape[0] * (fft_DAS_signals.shape[1] - 70))
    count = 0
    for time_idx in range(0, fft_DAS_signals.shape[0], 10):
        for channel_idx in range(60, fft_DAS_signals.shape[1] - 10, 1):
            # for channel_width_idx in range (-abs(channel_width) , channel_width,1 ):
            spectro_sample = np.abs(
                fft_DAS_signals[time_idx:time_idx + time_width, channel_idx - channel_width:channel_idx + channel_width,
                frequency_range[0]: frequency_range[1]])
            xmax, xmin = spectro_sample.max(), spectro_sample.min()
            # Normalizing the array 'x' using min-max scaling: (x - xmin) / (xmax - xmin)
            normalized_spectro = spectro_sample / spectro_sample.sum()
            # training_data.append(normalized_spectro)
            training_data[count, :, :, :] = normalized_spectro
            if time_idx > time_big_leak[0] and time_idx < time_big_leak[1] and channel_idx > channels_leak[
                0] and channel_idx < channels_leak[1]:
                training_label[count] = 1
            elif time_idx > time_small_leak[0] and time_idx < time_small_leak[1] and channel_idx > channels_leak[
                0] and channel_idx < channels_leak[1]:
                training_label[count] = 2
            else:
                training_label[count] = 0
            count += 1
        if time_idx % 100 == 0:
            print(time_idx)

    return training_data, training_label


def process_file_welch(dataArray, Fs: int, timeToProcess: int, overlap: bool):
    if overlap:
        overlapValue = (Fs * timeToProcess) / 2
    else:
        overlapValue = None

    fft_freq, fft_data = welch(dataArray, fs=Fs, window='blackman', \
                               nperseg=(Fs * timeToProcess), \
                               noverlap=overlapValue, axis=0, detrend='linear')

    phasedB = 20 * np.log10((np.sqrt(fft_data)) / 1e-6)

    return fft_freq, phasedB




