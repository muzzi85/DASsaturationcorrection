import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import h5py
# import numba as nb
import numpy as npt
import scipy.signal as signal
from datetime import datetime
import pathlib
from scipy.signal import welch, decimate


### data loading
#@nb.njit(parallel=True)
# def fft_channelwise(
#     arr,
#     fs):
#     num_rows, num_channels = arr.shape
#     time_intervals = num_rows // fs
#     fft_array = np.empty(
#         (time_intervals, num_channels, (fs // 2) + 1),
#         dtype=np.float32,
#     )
#     for channel_num in nb.prange(num_channels):
#         for i, time_ids in enumerate(range(0, num_rows, fs)):
#             X = np.fft.rfft(arr[time_ids : time_ids + fs, channel_num]).real
#
#             X = np.abs(X)
#             fft_array[i, channel_num] = X
#     return fft_array

# @nb.njit(parallel=False)
def cumsum_channelwise(data):
    result = np.empty_like(data)
    result[0] = data[0]
    for i in range(1, data.shape[0]):
        result[i] = result[i - 1] + data[i]
    return result

def sosfilt_channelwise(
    arr,
    sampling_frequency,
    filter_cutoff_frequency,
) :
    nyquist = sampling_frequency / 2
    normal_cutoff = filter_cutoff_frequency / nyquist
    sos = signal.butter(
        N=3,
        Wn=normal_cutoff,
        btype="high",
        output="sos",
        analog=False,
    )
    return signal.sosfilt(sos, arr, axis=0)

def process_file_welch(dataArray, Fs:int, timeToProcess: int, overlap: bool):
    fft_freq_array = np.empty(
        (dataArray.shape[0], (20_000 // 2) + 1),
        dtype=np.float32,
    )
    fft_data_array = np.empty(
        (dataArray.shape[0], (20_000 // 2) + 1, dataArray.shape[2]),
        dtype=np.float32,
    )
    
    if overlap:
        overlapValue = (Fs*timeToProcess)/2
    else:
        overlapValue = None

    for index, slice in enumerate(dataArray):
        fft_freq, fft_data = welch(slice, fs=Fs,  window='blackman', \
                                    nperseg=(Fs*timeToProcess), \
                                    noverlap=overlapValue, axis=0, detrend='linear')
        fft_freq_array[index] = fft_freq
        fft_data_array[index] = fft_data
    
    fft_data_array = np.transpose(fft_data_array, (0, 2, 1))
    fft_data_array = 20*np.log10((np.sqrt(fft_data_array))/1e-6)

    return fft_freq_array, fft_data_array

def sort_key(filename):
    base = filename.stem
    timestamp_str = base.split('_')[2]  # Assumes format 'ID_DATE_TIME.hdf5'
    return datetime.strptime(timestamp_str, '%H.%M.%S.%f')

def get_file_names(path):
    file_path = pathlib.Path(path)
    file_count = 0
    for item in file_path.iterdir():
        if item.is_file() and item.suffix == '.hdf5':
            if file_count == 0:
                file_names = [item]
            else:
                file_names.append(item)
            file_count = file_count + 1
    #print('[HDF5 Processing] Number of Files', file_count)

    file_names.sort(key=sort_key)

    return file_names

def load_hdf5_file(path, file_names, channels_to_load, fs=10_000):
    data_combined = np.zeros( (len(file_names), fs*10, channels_to_load[1]-channels_to_load[0]) )
    delta_data_combined = np.zeros( (len(file_names), fs*10, channels_to_load[1]-channels_to_load[0]) )

    for index, file in enumerate(file_names):
        print(index, file)
        with h5py.File(file, 'r') as hf:
            n1 = hf.get('DAS')
            data = np.array(n1)[:, channels_to_load[0]:channels_to_load[1]]


            data = data * (np.pi / 2**15)

            data1 = cumsum_channelwise(data)

            filter_cutoff_frequency = 5
            data1 = sosfilt_channelwise(data1, fs, filter_cutoff_frequency)
        data_combined[index] = data1
        delta_data_combined[index] = data

    return delta_data_combined, data_combined


### data plotting
def plot_range(data, start, end, offset, path_to_save=None, frequency_range=[0,1000], step=3):

    fontsize = 16
    linewidth = 3

    ncols = 8

    channels_leak1 = np.arange(start, end,step)
    nrows = (len(channels_leak1)//ncols)+1

    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6*ncols, 6*nrows), sharey=False)
    ax = ax.ravel()


    # Loop through each channel and plot the data_11
    for plot, channel in enumerate(channels_leak1):
        ax[plot].set_title(f'{channels_leak1[plot]+offset}', fontsize=fontsize)
        plot_aux_bkg = data[:, channels_leak1[plot]:channels_leak1[plot]+step, frequency_range[0]:frequency_range[1]].mean(0).mean(0) 
        ax[plot].plot(plot_aux_bkg, linewidth=linewidth, c='blue')

        ax[plot].tick_params(axis='both', labelsize=fontsize)

    ax[0].legend(fontsize=fontsize)

    plt.tight_layout()
    if path_to_save:
        fig.savefig(path_to_save)


def plot_range_td(data, offset, start, end, decimateFactor=10, path_to_save=None, step=3):
    data = decimate(data, decimateFactor, axis=0)

    fontsize = 16
    linewidth = 3

    ncols = 8

    channels_leak1 = np.arange(start, end,step)
    nrows = (len(channels_leak1)//ncols)+1

    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6*ncols, 6*nrows), sharey=False)
    ax = ax.ravel()
    # Loop through each channel and plot the data_11
    for plot, channel in enumerate(channels_leak1):
        ax[plot].set_title(f'{channels_leak1[plot]+offset}', fontsize=fontsize)
        ax[plot].plot(data[:,channel], c='blue')

        ax[plot].tick_params(axis='both', labelsize=fontsize)

    ax[0].legend(fontsize=fontsize)

    fig.tight_layout()
    if path_to_save:
        fig.savefig(path_to_save)

def spectrum_per_10_sec(data_fft_welch, start, path_to_save=None):
    channels_to_plot = np.arange(85,105,2)

    fontsize=14
    linewidth=2
    offset = start
    step = 2
    ncols = len(channels_to_plot)
    nrows = len(data_fft_welch)
    ncols, nrows

    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(4*ncols, 4*nrows), sharey=False)

    for row_idx, data in enumerate(data_fft_welch):
        for channel_idx in range(len(channels_to_plot)):
            ax[row_idx][channel_idx].set_title(f'{channels_to_plot[channel_idx]}', fontsize=fontsize)
            plot_aux = data[channels_to_plot[channel_idx]-offset:channels_to_plot[channel_idx]-offset+step, 200:700].mean(0) 
            ax[row_idx][channel_idx].plot(plot_aux, linewidth=linewidth, c='blue')
            ax[row_idx][channel_idx].tick_params(axis='both', labelsize=fontsize)

    fig.tight_layout()
    if path_to_save:
        fig.savefig(path_to_save)

def spectrogram_over_channels(data_fft_welch, data_freq, path_to_save):
    channel_count = data_fft_welch.shape[1]
    fontsize=14
    distance_axis = np.linspace(0, channel_count, channel_count)
    distance_axis.shape

    ncols = 6
    nrows = data_fft_welch.shape[0]//ncols+1

    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6*ncols, 6*nrows), sharey=False)
    ax = ax.ravel()

    for index, data in enumerate(data_fft_welch[:,:,0:1000]):
        ax[index].pcolormesh(distance_axis, data_freq[0, 0:1000], data.transpose(1,0), cmap='jet', shading='auto')
        ax[index].set_yscale('log')
        ax[index].set_ylim([10, 1000])
        ax[index].set_title(f'time slice (10 sec)={index}')
        ax[index].set_xlabel('Distance [channel]', fontsize=fontsize)
        ax[index].set_ylabel('Frequency [Hz]', fontsize=fontsize)
        ax[index].tick_params(axis='both', labelsize=fontsize)


    fig.tight_layout()
    if path_to_save:
        fig.savefig(path_to_save)

def spectrogram_over_time(data_fft_welch, data_freq, path_to_save):
    fontsize=14

    time_axis = np.linspace(0, data_fft_welch.shape[0], data_fft_welch.shape[0])
    time_axis.shape

    ncols = 6
    nrows = 186//ncols+1

    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6*ncols, 6*nrows), sharey=False)
    ax = ax.ravel()

    for index, data in enumerate(data_fft_welch.transpose(1,0,2)[:,:,0:1000]):
        ax[index].set_title(f'channel={index}')
        ax[index].set_xlabel('Time [s]', fontsize=fontsize)
        ax[index].set_ylabel('Frequency [Hz]', fontsize=fontsize)

        ax[index].pcolormesh(time_axis, data_freq[0,0:1000], data.transpose(1,0), cmap='jet', shading='auto')
        ax[index].set_yscale('log')
        ax[index].set_ylim([10, 1000])
        ax[index].tick_params(axis='both', labelsize=fontsize)

    fig.tight_layout()
    if path_to_save:
        fig.savefig(path_to_save)


def leak_selection_heatmap(data_fft_welch, leak_start_c, leak_end_c, leak_start_t, leak_end_t, locations, offset, start, end, path_to_save=None):
    vmax = np.percentile(data_fft_welch[:,locations['c1']:locations['c7'],200:500].sum(2), q=95)

    ncols=1
    nrows=1

    linewidth=1
    labelsize=12

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15,8*nrows))
    # axs = axs.ravel()

    axs.pcolormesh(data_fft_welch[:,:,200:500].sum(2), shading='auto', vmax=vmax)
    axs.set_xlabel('Channel')
    axs.set_ylabel('Time')
    axs.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=40))

    xticks = axs.get_xticks()
    xticks_with_offset = xticks + offset
    axs.set_xticks(xticks)
    axs.set_xticklabels(xticks_with_offset.astype(int));


    axs.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=30))

    for name, chamber in locations.items():
        if start <= chamber <= end:
            axs.axvline(x=chamber-offset, color='white', linestyle='-', linewidth=linewidth)
            axs.text(chamber-offset + 0.1, 0.95, name, transform=axs.get_xaxis_transform(), color='white', ha='left', va='top', fontsize=labelsize)

    axs.axvline(x=leak_start_c-offset, color='black', linestyle='--', linewidth=linewidth)
    axs.axvline(x=leak_end_c-offset, color='black', linestyle='--', linewidth=linewidth)
    axs.axhline(y=leak_start_t, color='black', linestyle='--', linewidth=linewidth)
    axs.axhline(y=leak_end_t, color='black', linestyle='--', linewidth=linewidth)


    plt.grid(alpha=0.2)
    fig.tight_layout()
    if path_to_save:
        fig.savefig(path_to_save)

def final_spectrums(final_array, path_to_save=None):
    channels_to_plot = np.arange(0,final_array.shape[1],1)

    fontsize=14
    linewidth=2
    ncols = len(channels_to_plot)
    nrows = len(final_array)

    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(4*ncols, 4*nrows), sharey=False)

    for row_idx, data in enumerate(final_array):
        for channel_idx in range(len(channels_to_plot)):
            plot_aux = data[channels_to_plot[channel_idx], :]
            ax[row_idx][channel_idx].set_title(f'variance={np.var(plot_aux):.2f}', fontsize=fontsize)
            ax[row_idx][channel_idx].plot(plot_aux, linewidth=linewidth, c='blue')
            ax[row_idx][channel_idx].tick_params(axis='both', labelsize=fontsize)

    fig.tight_layout()
    if path_to_save:
        fig.savefig(path_to_save)

### misc
def reshape_with_padding_or_trimming(data, chunk_size, pad_mode='edge', threshold=0.7):
    """
    Reshape a 3D numpy array into a 4D array by grouping the first axis into chunks of a specified size.
    
    The function checks the remainder of the first axis when divided by chunk_size:
      - If the number of missing samples to reach the next full chunk is less than threshold * chunk_size,
        the data is padded along axis 0 using the specified pad_mode.
      - Otherwise, the extra samples are trimmed off so that only complete chunks remain.
    
    Parameters:
        data (np.ndarray): A 3D numpy array with shape (n_samples, dim2, dim3).
        chunk_size (int): The size of each chunk for the new second axis (e.g., 100).
        pad_mode (str): Padding mode for np.pad (default is 'edge').
        threshold (float): The fraction of the chunk size to use as the threshold for padding (default is 0.7).
    
    Returns:
        np.ndarray: A reshaped 4D array with shape (n_chunks, chunk_size, dim2, dim3).
    """
    n_samples = data.shape[0]
    remainder = n_samples % chunk_size
    missing = chunk_size - remainder if remainder != 0 else 0

    if remainder != 0:
        if missing < threshold * chunk_size:
            # Pad along axis 0 to complete the next chunk.
            pad_width = ((0, missing),) + ((0, 0),) * (data.ndim - 1)
            data = np.pad(data, pad_width=pad_width, mode=pad_mode)
            print(f"Padded with {missing} samples to reach {data.shape[0]} samples.")
        else:
            # Trim the data to remove extra samples.
            data = data[:n_samples - remainder]
            print(f"Trimmed off {remainder} samples, leaving {data.shape[0]} samples.")

    # Reshape the data into 4D.
    new_shape = (-1, chunk_size) + data.shape[1:]
    return data.reshape(new_shape)
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def preprocessing(fft_DAS):
    DAS_channels = fft_DAS.shape[1]
    Post_procssed = np.zeros((DAS_channels, 1, 500, 1))

    for channel_idx in range(5, DAS_channels, 1):
        # print(channel_idx)
        sample_time = np.abs(fft_DAS[:, channel_idx:channel_idx + 5, :]).sum(0).sum(0)
        curved_data_time = NormalizeData(sample_time)
        curved_data_time = np.expand_dims(curved_data_time, axis=0)
        curved_data_time = np.expand_dims(curved_data_time, axis=-1)
        Post_procssed[channel_idx] = curved_data_time
    return Post_procssed