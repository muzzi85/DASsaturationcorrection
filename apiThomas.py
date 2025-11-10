import h5py
import numpy as np
import os

# just config
path_to_file = r"Y:\APSensing Ramp Data\0000005762_2023-12-18_14.23.08.64718.hdf5"
dataset_name = "DAS"  # Change if your dataset is named differently
rows_per_chunk = 1000  # Adjust based on available RAM
cols_slice = slice(400, 440)  # Adjust columns if needed (slice or None for all)
output_path = "recovered_das.npy"  # or change to .csv later

# open the file
with h5py.File(path_to_file, 'r') as hf:
    if dataset_name not in hf:
        raise KeyError(f"Dataset '{dataset_name}' not found in the file.")

    n1 = hf[dataset_name]
    print(f"Dataset '{dataset_name}' opened successfully.")
    print(f"Shape: {n1.shape}, dtype: {n1.dtype}")

    rows, cols = n1.shape
    recovered_blocks = []
    corrupted_blocks = []

    # iterate over blocks
    for i in range(0, rows, rows_per_chunk):
        j = min(i + rows_per_chunk, rows)
        try:
            print(f"Reading rows {i}:{j} ...", end="")
            block = n1[i:j, cols_slice]
            recovered_blocks.append(block)
            print(" ✅ OK")
        except OSError:
            print(" ⚠️ Corrupted block — skipping.")
            corrupted_blocks.append((i, j))
            continue

# combine success blocks
if recovered_blocks:
    recovered_data = np.vstack(recovered_blocks)
    print(f"\n✅ Recovered {recovered_data.shape[0]} rows successfully.")

    # uncomment if you want to save blocks
    # np.save(output_path, recovered_data)
    # print(f"Data saved to: {os.path.abspath(output_path)}")
else:
    print("\n❌ No readable data recovered.")

# report corrupted blocks
if corrupted_blocks:
    print("\n⚠️ Corrupted regions found:")
    for (start, end) in corrupted_blocks:
        print(f" - Rows {start}:{end}")
else:
    print("\n✅ No corrupted blocks detected.")




## loading das data
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from math import log
import random
from ZstdArrayHandler import ZstdArrayHandler
handler = ZstdArrayHandler(compression_level=3)
import h5py

def load_single_DAS_file(file_name):
    hf = h5py.File(file_name, 'r')
    n1 = hf.get('DAS')
    n2 = np.array(n1)
    n2 = n2 * (np.pi / 2 ** 15)
    #print(f'[HDF5 Processing] Integrate')
    n22 = np.cumsum(n2, axis=0)
    return n2, n22

path_to_files = f'Z:/1_ds_data_organization_morocco/1_hdf5/1_with_flow/11_06_2024/0000470390_2024-06-11_22.59.58.52288.hdf5'#120000850-144500850_Feb-02'#f'Z:/1_ds_data_organization_morocco/1_hdf5/1_with_flow/11_06'#120000850-144500850_Feb-02
path_to_files = f"D:/Saturation/0000470390_2024-06-11_22.59.58.52288.hdf5"

path_to_files = "Y:/APSensing Ramp Data/0000005762_2023-12-18_14.23.08.64718.hdf5"

hf = h5py.File(path_to_files, 'r')
n1 = hf.get('DAS')

delta_phase_data, phase_data = load_single_DAS_file(path_to_files)

def saturation_metric_accel(diff_phase_sample: np.ndarray, limit: float = 1.4 * np.pi) -> float:
    """Calculate phase acceleration metric (second difference of phase) for a given sample of diff_phase data.

    Uses all values in the array for the statistics of the metrics.

    Parameters
    ----------
    diff_phase_sample : np.ndarray
        Differential phase values in rad/trace
    limit : float, optional
        Limit for second phase difference, by default 1.4*np.pi

    Returns
    -------
    float
        Acceleration metric for phase sample
    """
    accel_phase = np.diff(diff_phase_sample, axis=0)
    exceed = np.abs(accel_phase) > limit
    return np.count_nonzero(exceed), exceed


cc, exceed = saturation_metric_accel((n2[0:, 33]))
delta_phi_corr, phi_corr, phi_diff = unwrap_delta_phase(n2[200030:200030+20000,24],n_history=2) # plt.plot((n2[ 200030:200030+20,24]))
delta_phi_corrn, phi_corrn, phi_diffn = unwrap_delta_phase(n2[100030:100030+20000, 24],n_history=2) # plt.plot((n2[ 200030:200030+20,24]))
delta_phi_corrn, phi_corrn, phi_diffn = unwrap_delta_phase(n2[20200:300030, 24],n_history=3) # plt.plot((n2[ 200030:200030+20,24]))

unwrapped, corrected, k_map = delta_phase_unwrap_v2(np.cumsum(n2[0:, :], axis=0).T,
                          window_k=4,
                          median_prefilter=False,
                          median_kernel=3,
                          spatial_iter=2)



## filter
import scipy.signal as signal

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

data_filtered = sosfilt_channelwise(delta_phi_corrn[ 0:250200+100], 40000, 25)


## fft

import numpy as np

import numpy as np

def windowed_fft(signal, fs, window_size, overlap, window_type='hanning'):
    """
    Compute windowed FFT with overlap.

    Parameters
    ----------
    signal : np.ndarray
        Input 1D signal.
    fs : float
        Sampling frequency (Hz).
    window_size : int
        Number of samples per window (FFT length).
    overlap : float
        Overlap fraction (0.0–1.0), e.g. 0.5 for 50% overlap.
    window_type : str
        Window function: 'hanning', 'hamming', 'blackman', etc.

    Returns
    -------
    freqs : np.ndarray
        Frequency axis (Hz).
    times : np.ndarray
        Time axis (seconds).
    magnitude_spectrogram : 2D np.ndarray
        Magnitude of the FFT (shape: [n_freqs, n_segments]).
    """
    window_size = int(window_size)
    step = int(window_size * (1 - overlap))
    if step <= 0:
        raise ValueError("Overlap too large — must be less than 1.0")

    n_segments = (len(signal) - window_size) // step + 1

    # Select window
    if hasattr(np, window_type):
        window = getattr(np, window_type)(window_size)
    else:
        raise ValueError(f"Unknown window type: {window_type}")

    freqs = np.fft.rfftfreq(window_size, d=1/fs)
    magnitude_spectrogram = []

    for i in range(n_segments):
        start = i * step
        segment = signal[start:start + window_size]

        # Apply window
        windowed = segment * window

        # Compute one-sided FFT
        fft_result = np.fft.rfft(windowed)
        magnitude = np.abs(fft_result) * 2 / window_size
        magnitude_spectrogram.append(magnitude)

    magnitude_spectrogram = np.array(magnitude_spectrogram).T
    times = np.arange(n_segments) * step / fs

    return freqs, times, magnitude_spectrogram
fs = 10000  # sampling frequency in Hz
# signal_wronge = (n2[ 250200+20200:250200+20200+200,24]) # example 50 Hz sine wave
# signal_corrected = (delta_phi_corrn[ 250200:250200+1000])) # example 50 Hz sine wave
signal_wronge = (n2[ 20200:250200+20200+(fs*1),24]) # example 50 Hz sine wave
signal_corrected = (delta_phi_corrn[:250200+(fs*1)]) # example 50 Hz sine wave

signal_wronge = signal_wronge - np.mean(signal_wronge)
signal_corrected = signal_corrected - np.mean(signal_corrected)
## filter

from scipy.signal import butter, filtfilt

def bandpass_filter(signal, fs, f0, bw=20):
    low = (f0 - bw/2) / (fs/2)
    high = (f0 + bw/2) / (fs/2)
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, signal)

signal_wronge = bandpass_filter(signal_wronge, fs, f0=500, bw=30)
signal_corrected = bandpass_filter(signal_corrected, fs, f0=500, bw=30)


freqs, times, magwronge = windowed_fft(signal_wronge, fs=fs, window_size=fs/2,overlap=0.5)
freqs, times, magcorrected = windowed_fft(signal_corrected,  fs=fs, window_size=fs/2,overlap=0.5)

import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.pcolormesh(times, freqs, 20*np.log10(magwronge + 1e-12), shading='auto')
plt.title("Windowed FFT (Spectrogram) - magwronge")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.colorbar(label="Magnitude [dB]")
plt.show()

import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.pcolormesh(times, freqs, 20*np.log10(magcorrected + 1e-12), shading='auto')
plt.title("Windowed FFT (Spectrogram) - magcorrected")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.colorbar(label="Magnitude [dB]")
plt.show()


### load Morocow data

import glob
txtfiles = []
path_to_files = f'Z:/1_ds_data_organization_morocco/1_hdf5/1_with_flow/11_06_2024'#120000850-144500850_Feb-02 # Y:\5_data_organisation_mc\1_hdf5\3_flow_unknown\100sec_18-Aug_muzzi

import glob

for file in glob.glob("Z:/1_ds_data_organization_morocco/1_hdf5/1_with_flow/11_06_2024/*.hdf5"):
    print(file)
    txtfiles.append(file)
time_interval=[]
txtfiles.sort()
file = "Z:/1_ds_data_organization_morocco/1_hdf5/1_with_flow/11_06_2024/0000470390_2024-06-11_22.59.58.52288.hdf5"
delta_phase_data, phase_data = load_single_DAS_file(file)
file = "Z:/1_ds_data_organization_morocco/1_hdf5/1_with_flow/11_06_2024/0000470391_2024-06-11_23.00.08.52288.hdf5"
delta_phase_data1, phase_data1 = load_single_DAS_file(file)
file = "Z:/1_ds_data_organization_morocco/1_hdf5/1_with_flow/11_06_2024/0000470392_2024-06-11_23.00.18.52288.hdf5"
delta_phase_data2, phase_data2 = load_single_DAS_file(file)

ch = 450
delta_phi_corrn, phi_corrn, phi_diffn = unwrap_delta_phase(delta_phase_data[:, ch], n_history=3) # plt.plot((n2[ 200030:200030+20,24]))

plt.figure()
plt.plot(delta_phi_corrn[0:1000])
plt.xlabel("Time samples")
plt.ylabel("Delta phase corrected")

plt.figure()
plt.plot(delta_phase_data[0:1000, ch])
plt.xlabel("Time samples")
plt.ylabel("Delta phase ")

plt.figure()
plt.plot(delta_phi_corrn)
plt.figure()
plt.plot(delta_phase_data[:, 450])


plt.figure()
plt.plot(np.cumsum(delta_phi_corrn[0:]))
plt.figure()
plt.plot(np.cumsum(delta_phase_data[0:, 450]))

plt.figure()
plt.plot(np.cumsum(delta_phi_corrn[0:5000]))
plt.figure()
plt.plot(np.cumsum(delta_phase_data[0:5000, 450]))

## zac and Harry data
import matplotlib.pyplot as plt
import numpy as np

def wrap(value: float) -> float:
    """Wrap a value between +-π"""

    return (value + np.pi) % (np.pi * 2) - np.pi

def generate_sine_wave(
    duration: float,
    sample_rate: int,
    frequency: float,
    start_amplitude: float,
    stop_amplitude: float,
):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    amplitude = np.linspace(start_amplitude, stop_amplitude, len(t))
    signal = amplitude * np.sin(2 * np.pi * frequency * t)

    return t, signal


duration = 20.0
sample_rate = 10_000  # Hz
frequency = 60  # Hz
start_amplitude = 0.1
stop_amplitude = 400

# Generate the sine wave
t, signal = generate_sine_wave(
    duration, sample_rate, frequency, start_amplitude, stop_amplitude
)

print(np.max(np.diff(signal)))
print(np.min(np.diff(signal)))

x = wrap(signal)
y = wrap(np.diff(x))
z = np.cumsum(y)

figure, axes = plt.subplots(4, sharex=True)

axes[0].set_title("Original Signal")
axes[0].plot(t, signal)

axes[1].set_title("Wrapped Phase")
axes[1].plot(t, x)

axes[2].set_title("Wrapped Differential Phase")
axes[2].plot(t[:-1], y)

axes[3].set_title("Reconstructed Phase")
axes[3].plot(t[:-1], z)

plt.show()

delta_phi_corr, phi_corr, phi_diff = unwrap_delta_phase(y)

figure, axes = plt.subplots(4, sharex=True)

indexx = 10000
axes[0].set_title("Wrapped Differential Phase original")
axes[0].plot(t[indexx:indexx+100000], y[indexx:indexx+100000])

axes[1].set_title("Wrapped Differential Phase corrected")
axes[1].plot(t[indexx:indexx+100000], delta_phi_corr[indexx:indexx+100000])
plt.show()
zz = np.cumsum(delta_phi_corr[indexx:indexx+100000])
axes[3].set_title("Reconstructed Phase corrected")
axes[3].plot(t[indexx:indexx+100000], zz)

axes[2].set_title("Reconstructed Phase original")
axes[2].plot(t[indexx:indexx+100000], z[indexx:indexx+100000])

### fft for Moro

signal_wronge = (delta_phase_data[ :,450]) # example 50 Hz sine wave
signal_corrected = (delta_phi_corrn) # example 50 Hz sine wave

freqs, times, magwronge = windowed_fft(signal_wronge, fs=fs, window_size=fs/2,overlap=0.5)
freqs, times, magcorrected = windowed_fft(signal_corrected,  fs=fs, window_size=fs/2,overlap=0.5)

import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.pcolormesh(times, freqs, 20*np.log10(magwronge + 1e-12), shading='auto')
plt.title("Windowed FFT (Spectrogram) - magwronge")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.colorbar(label="Magnitude [dB]")
plt.show()

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.pcolormesh(times, freqs, 20*np.log10(magcorrected + 1e-12), shading='auto')
plt.title("Windowed FFT (Spectrogram) - magcorrected")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.colorbar(label="Magnitude [dB]")
plt.show()




## plot ks over phase
import matplotlib.pyplot as plt
import numpy as np
ch = 24
# Example: adjust according to your variable names
x = np.arange(0, 320000+5000)
y1 = np.cumsum(n2[0:320000+5000, ch])
y2 = k_map[ch, 0:320000+5000] * 1  # scaling if needed

fig, ax1 = plt.subplots(figsize=(10, 5))

# --- First dataset (left y-axis) ---
color1 = 'tab:blue'
ax1.plot(x, y1, color=color1, label='Cumulative delta phase')
ax1.set_xlabel('Time sample')
ax1.set_ylabel('Cumulative of Delta Phase', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)

# --- Second dataset (right y-axis) ---
ax2 = ax1.twinx()
color2 = 'tab:orange'
ax2.plot(x, y2, color=color2, label='2pi-rounds')
ax2.set_ylabel('2-pi round suggested', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

# --- Combine legends from both axes ---
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')

# --- Optional cosmetics ---
ax1.grid(True, which='major', linestyle='--', alpha=0.5)
plt.title('Comparison of delta phase and 2-pi rounds at Channel'+str(ch))
fig.tight_layout()

plt.show()


###
plt.plot(np.cumsum(n2[0:320000+5000,24]))
plt.plot((k_map[24, 0:320000+5000])*1)


plt.plot(np.cumsum(n2[0:, 33]))
plt.plot(exceed*100, 'k')
plt.xlabel("samples")
plt.ylabel("cumsum")


plt.plot(n2[0:1000, 33], 'r')
plt.plot(delta_phi_corr, 'g')
plt.plot(exceed, 'k')



plt.plot(n2[:],'r')
plt.plot(delta_phi_corr[:],'g')
plt.plot(exceed[:],'k')
