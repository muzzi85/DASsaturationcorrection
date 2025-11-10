


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
    n22 = np.cumsum(n2,axis=0)
    return n2, n22

path_to_files = f'Z:/1_ds_data_organization_morocco/1_hdf5/1_with_flow/11_06_2024/0000470390_2024-06-11_22.59.58.52288.hdf5'#120000850-144500850_Feb-02'#f'Z:/1_ds_data_organization_morocco/1_hdf5/1_with_flow/11_06'#120000850-144500850_Feb-02
path_to_files=f"D:/Saturation/0000470390_2024-06-11_22.59.58.52288.hdf5"

## 0.
path_to_files="Z:/2_data_organisation_ramsbrook/1_hdf5/set3/flow_tests_0_3-1_5_04_02_2025/0.0/0001053298_2025-02-04_13.39.23.90621.hdf5"
delta_phase_data_0, phase_data_0 = load_single_DAS_file(path_to_files)

## 0.3
path_to_files="Z:/2_data_organisation_ramsbrook/1_hdf5/set3/flow_tests_0_3-1_5_04_02_2025/0.3/0001053311_2025-02-04_13.41.33.90621.hdf5"
delta_phase_data_03, phase_data_03 = load_single_DAS_file(path_to_files)

## 0.6
path_to_files="Z:/2_data_organisation_ramsbrook/1_hdf5/set3/flow_tests_0_3-1_5_04_02_2025/0.6/0001053356_2025-02-04_13.49.03.90621.hdf5"
delta_phase_data_06, phase_data_06 = load_single_DAS_file(path_to_files)

## 0.9
path_to_files="Z:/2_data_organisation_ramsbrook/1_hdf5/set3/flow_tests_0_3-1_5_04_02_2025/0.9/0001053386_2025-02-04_13.54.03.90621.hdf5"
delta_phase_data_09, phase_data_09 = load_single_DAS_file(path_to_files)

## 1.2
path_to_files="Z:/2_data_organisation_ramsbrook/1_hdf5/set3/flow_tests_0_3-1_5_04_02_2025/1.2/0001053421_2025-02-04_13.59.53.90621.hdf5"
delta_phase_data_12, phase_data_12 = load_single_DAS_file(path_to_files)

## 1.5
path_to_files="Z:/2_data_organisation_ramsbrook/1_hdf5/set3/flow_tests_0_3-1_5_04_02_2025/1.5/0001053454_2025-02-04_14.05.23.90621.hdf5"
delta_phase_data_15, phase_data_15 = load_single_DAS_file(path_to_files)

path_to_files="Z:/2_data_organisation_ramsbrook/1_hdf5/set3/flow_tests_0_3-1_5_04_02_2025/0.0/0001053298_2025-02-04_13.39.23.90621.hdf5"
delta_phase_data_0, phase_data_0 = load_single_DAS_file(path_to_files)


## delta phase per flow

## steel
DP0 = delta_phase_data_0[0:200000, 94]
DP03 = delta_phase_data_03[0:200000, 94]
DP06 = delta_phase_data_06[0:200000, 94]
DP09 = delta_phase_data_09[0:200000, 94]
DP12 = delta_phase_data_12[0:200000, 94]
DP15 = delta_phase_data_15[0:200000, 94]

## steel
DP0 = delta_phase_data_0[0:200000, 110]
DP03 = delta_phase_data_03[0:200000, 110]
DP06 = delta_phase_data_06[0:200000, 110]
DP09 = delta_phase_data_09[0:200000, 110]
DP12 = delta_phase_data_12[0:200000, 110]
DP15 = delta_phase_data_15[0:200000, 110]


### plot energy per 0.25 sec per flow condition using different frequency bins with sepepration
import numpy as np
import matplotlib.pyplot as plt

fs = 20000  # Sampling frequency
window_size = int(0.25 * fs)  # 0.25 s windows
freq_bins = [10, 50, 100, 250, 500, 1000]  # Hz

# Example datasets
datasets = {
    "0 m/s": DP0,
    "0.3 m/s": DP03,
    "0.6 m/s": DP06,
    "0.9 m/s": DP09,
    "1.2 m/s": DP12,
    "1.5 m/s": DP15
}

n_flows = len(datasets)
n_bins = len(freq_bins)
energies_per_flow_bin = np.zeros((n_flows, n_bins))  # mean energy per flow and bin
energies_all_windows = []  # for Fisher calculation

# Compute energies per window
for i, (label, data) in enumerate(datasets.items()):
    num_windows = len(data) // window_size
    energies_windows = np.zeros((num_windows, n_bins))

    for w in range(num_windows):
        window = data[w * window_size:(w + 1) * window_size]
        A = np.fft.rfft(window)
        freqs = np.fft.rfftfreq(len(window), 1/fs)
        PSD = (np.abs(A) ** 2) / (fs * len(window))  # linear scale

        for j, fmax in enumerate(freq_bins):
            idx = freqs <= fmax
            energy = np.sum(PSD[idx])
            energy_dB = 10 * np.log10(energy / 1e-12)
            energies_windows[w, j] = energy_dB

    energies_per_flow_bin[i, :] = energies_windows.mean(axis=0)
    energies_all_windows.append(energies_windows)

# Compute Fisher score per frequency bin
fisher_scores = np.zeros(n_bins)

for j in range(n_bins):
    numerator = 0
    denominator = 0
    for i in range(n_flows - 1):
        mu_i = energies_all_windows[i][:, j].mean()
        mu_next = energies_all_windows[i+1][:, j].mean()
        sigma_i2 = energies_all_windows[i][:, j].var()
        sigma_next2 = energies_all_windows[i+1][:, j].var()

        numerator += (mu_next - mu_i) ** 2
        denominator += sigma_i2 + sigma_next2

    fisher_scores[j] = numerator / denominator

best_bin_idx = np.argmax(fisher_scores)
best_bin_freq = freq_bins[best_bin_idx]
print(f"Best frequency range based on Fisher score: 0-{best_bin_freq} Hz")

# Plot energy vs flow with Fisher scores
plt.figure(figsize=(12,8))
for j, fmax in enumerate(freq_bins):
    plt.plot(list(datasets.keys()), energies_per_flow_bin[:, j], '-o', label=f'0-{fmax} Hz\nF={fisher_scores[j]:.2f}')

plt.xlabel("Flow [m/s]", fontsize=14)
plt.ylabel("Energy [dB rad² scale]", fontsize=14)
plt.title("Average Energy per Frequency Band with Fisher Scores", fontsize=16)
plt.grid(True, ls='--', alpha=0.6)
plt.legend(title="Frequency range", fontsize=12)
plt.tight_layout()
plt.savefig("dgd.png", dpi=500, bbox_inches="tight")
plt.show()


### plot energy per 0.25 sec per flow condition

import numpy as np
import matplotlib.pyplot as plt

fs = 20000  # sampling frequency
window_size = int(0.25 * fs)  # 0.25 sec = 5000 samples
freq_max = 1000  # Hz

# --- Example datasets ---
# Replace these with your actual delta_phase arrays
N = 20000
t = np.arange(N) / fs

datasets = {
    "0 m/s": DP0,
    "0.3 m/s": DP03,
    "0.6 m/s": DP06,
    "0.9 m/s": DP09,
    "1.2 m/s": DP12,
    "1.5 m/s": DP15
}

plt.figure(figsize=(12,6))

# Define line styles for each flow
line_styles = ['-', '--', '-.', ':', (0, (3,1,1,1)), (0, (5,1))]  # 6 different styles
final_energy = np.zeros((40, 6))
count = 0
for (label, data), ls in zip(datasets.items(), line_styles):
    num_windows = len(data) // window_size
    accumulated_energy_dB = []

    for i in range(num_windows):
        window = data[i*window_size:(i+1)*window_size]
        # FFT and PSD
        A = np.fft.rfft(window)
        freqs = np.fft.rfftfreq(len(window), 1/fs)
        PSD = (np.abs(A)**2) / (fs * len(window))  # rad²/Hz

        # Select 0–1000 Hz
        idx = freqs <= freq_max
        energy = np.sum(PSD[idx])
        energy = np.sum(PSD[0:15])

        energy_dB = 10 * np.log10(energy / 1e-12)
        accumulated_energy_dB.append(energy_dB)
        final_energy[i, count] = energy_dB

    count+=1
    # X-axis: cumulative time in windows
    x = np.arange(1, num_windows+1) * 0.25  # seconds
    plt.plot(x, accumulated_energy_dB, lw=2, linestyle=ls, label=label)

plt.xlabel("Time [s] (accumulated per 0.25 s)", fontsize=14)
plt.ylabel("Energy [dB re rad²] (0–1000 Hz band)", fontsize=14)
# plt.title("Accumulated Delta Phase Energy vs Time for Different Flows", fontsize=16)
plt.grid(True, ls='--', alpha=0.7)
plt.legend(loc='upper left')
plt.savefig("dfsdf.png", dpi=500, bbox_inches="tight")
plt.tight_layout()

### cal curve

import numpy as np
import matplotlib.pyplot as plt

# Flow rates and mean energy
flows = np.array([0, 0.3, 0.6, 0.9, 1.2, 1.5])
mean_energy_dB = np.mean(final_energy, axis=0)

# Fit 2nd order polynomial
coeffs = np.polyfit(flows, mean_energy_dB, 2)
poly = np.poly1d(coeffs)

# Smooth curve for visualization
flow_smooth = np.linspace(0, 1.5, 100)
energy_fit = poly(flow_smooth)

# Plot
plt.figure(figsize=(8,6))
plt.plot(flows, mean_energy_dB, 'o', label='Measured Energy', lw=2)
plt.plot(flow_smooth, energy_fit, '-', label='2nd-order Poly Fit', lw=2)
plt.xlabel("Flow rate [m/s]", fontsize=14)
plt.ylabel("Mean Energy [dB re rad²] over 10 seconds", fontsize=14)
plt.title("Calibration Curve with Polynomial Fit", fontsize=16)
plt.grid(True, ls='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

### MSE

import numpy as np
import matplotlib.pyplot as plt

# --- Data ---
flows = np.array([0, 0.3, 0.6, 0.9, 1.2, 1.5])
mean_energy_dB = np.mean(final_energy, axis=0)

# --- Fit 2nd-order polynomial ---
coeffs = np.polyfit(flows, mean_energy_dB, 2)
poly = np.poly1d(coeffs)

# Smooth curve for visualization
flow_smooth = np.linspace(0, 1.5, 100)
energy_fit = poly(flow_smooth)

# --- Plot ---
plt.figure(figsize=(12,9))
plt.plot(flows, mean_energy_dB, 'o', label='Measured Energy', lw=6)
plt.plot(flow_smooth, energy_fit, '-', label='2nd-order Poly Fit', lw=6)

# --- Draw vertical arrows showing error ---
for x, y in zip(flows, mean_energy_dB):
    y_fit = poly(x)
    plt.annotate(
        '',
        xy=(x, y_fit),
        xytext=(x, y),
        arrowprops=dict(arrowstyle='<->', color='red', lw=1.5)
    )

# --- Compute MSE ---
mse = np.mean((mean_energy_dB - poly(flows))**2)

# --- Compute percentage error per flow ---
perc_error = np.abs(mean_energy_dB - poly(flows)) / np.abs(poly(flows)) * 100
avg_perc_error = np.mean(perc_error)

y_min = min(mean_energy_dB)
y_max = max(mean_energy_dB)
y_pos = y_min + 0.05 * (y_max - y_min)  # 5% above the min

plt.text(1.01, y_pos,
         f'MSE = {mse:.3f} dB²\nAvg. Flow Error = {avg_perc_error:.2f} %',
         color='red', fontsize=16)

# --- Labels and styling ---
plt.xlabel("Flow rate [m/s]", fontsize=16)
plt.ylabel("Mean Energy [dB re rad²] over 10 seconds", fontsize=16)
# plt.title("Calibration Curve with Polynomial Fit", fontsize=18)
plt.grid(True, ls='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

plt.savefig("differentflow.png", dpi=500, bbox_inches="tight")
plt.show()

### flow estimation

import numpy as np

def estimate_flow(delta_phase_data, fs, window_sec, poly_fit, freq_max=1000):
    """
    Estimate flow from delta phase data using calibration polynomial.

    Parameters
    ----------
    delta_phase_data : ndarray, shape (n_samples, n_channels)
        DAS delta phase data.
    fs : int
        Sampling frequency in Hz.
    window_sec : float
        Window length in seconds for energy calculation.
    poly_fit : np.poly1d
        Calibration polynomial (energy in dB -> flow rate).
    freq_max : float
        Maximum frequency for PSD integration (Hz).

    Returns
    -------
    flow_estimates : ndarray, shape (n_windows, n_channels)
        Estimated flow per time window per channel.
    """
    n_samples, n_channels = delta_phase_data.shape
    window_size = int(window_sec * fs)
    n_windows = n_samples // window_size

    flow_estimates = np.zeros((n_windows, n_channels))

    for ch in range(n_channels):
        data_ch = delta_phase_data[:, ch]

        for w in range(n_windows):
            window = data_ch[w * window_size:(w + 1) * window_size]
            # FFT and PSD
            A = np.fft.rfft(window)
            freqs = np.fft.rfftfreq(len(window), 1 / fs)
            PSD = (np.abs(A) ** 2) / (fs * len(window))  # rad²/Hz

            # Integrate energy in 0–freq_max
            idx = freqs <= freq_max
            #energy = np.sum(PSD[idx])
            energy = np.sum(PSD[0:15])

            energy_dB = 10 * np.log10(energy / 1e-12)

            # Estimate flow using polynomial calibration
            # Use numpy roots to invert poly: poly(flow) - energy_dB = 0
            # --- Estimate flow using polynomial calibration ---
            # Solve poly(flow) - energy_dB = 0
            roots = np.roots(poly_fit - energy_dB)

            # Keep only real roots
            real_roots = roots[np.isreal(roots)].real

            # Keep only positive roots
            valid_roots = real_roots[real_roots >= 0]

            if len(valid_roots) == 0:
                flow_estimates[w, ch] = 0  # no valid solution
            else:
                # pick smallest positive root (or could pick closest to previous)
                flow_estimates[w, ch] = np.min(valid_roots)
    return flow_estimates


# Example parameters

data_to_test = np.concatenate([delta_phase_data_0*0.99,delta_phase_data_03 * 1.3,delta_phase_data_06*1.1, delta_phase_data_09,delta_phase_data_12*1.2,delta_phase_data_15*1.2])

fs = 20000
window_sec = 0.25  # 0.25 s window
flow_array = estimate_flow(data_to_test[:, 90:130], fs, window_sec, poly)
flow_array[flow_array>5]= 0
print(flow_array.shape)  # (n_windows, n_channels)
plt.imshow(flow_array, aspect="auto")
plt.colorbar()

plt.figure()
plt.plot(flow_array.mean(1))


### nice visulas of flow estimated

import numpy as np
import matplotlib.pyplot as plt

# --- Setup (replace with your actual flow_array) ---
# flow_array = np.random.rand(240, 40) * 1.5

fs = 20000
window_sec = 0.25
n_windows, n_channels = flow_array.shape

# Axes
time = np.arange(n_windows) * window_sec          # seconds
distance = np.arange(n_channels) * 2.45           # meters along pipe

# Flow test setup
flow_periods = [0, 0.3, 0.6, 0.9, 1.2, 1.5]       # m/s
seconds_per_flow = 10                             # each flow = 10 s
windows_per_period = int(seconds_per_flow / window_sec)  # 40 windows per flow

# --- Create figure with 2 subplots (heatmap + mean curve) ---
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 8), sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
def smooth_image(image, method='gaussian', kernel_size=100, sigma=2.0):
    """
    Smooths an image (NumPy array) using different smoothing methods.

    Parameters
    ----------
    image : np.ndarray
        Input 2D (grayscale) or 3D (color) image array.
    method : str
        One of ['gaussian', 'average', 'median'].
    kernel_size : int
        Size of smoothing window (odd number > 1).
    sigma : float
        Standard deviation for Gaussian smoothing.

    Returns
    -------
    smoothed : np.ndarray
        Smoothed image of same shape as input.
    """

    if method == 'gaussian':
        smoothed = gaussian_filter(image, sigma=sigma)

    elif method == 'average':
        smoothed = uniform_filter(image, size=kernel_size)

    elif method == 'median':
        smoothed = median_filter(image, size=kernel_size)

    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'gaussian', 'average', or 'median'.")

    return smoothed
import matplotlib.pyplot as plt
# Example noisy image
smoothed_gauss = smooth_image(flow_array, method='gaussian', sigma=6)
# --- (1) Flow image map ---
im = ax1.imshow(
    smoothed_gauss.T,
    aspect='auto',
    origin='lower',
    extent=[time[0], time[-1] + window_sec, distance[0], distance[-1]],
    cmap='viridis'
)

# Colorbar — positioned *underneath* to preserve horizontal alignment
cbar = fig.colorbar(im, ax=ax1, orientation='horizontal', pad=0.15, fraction=0.05)
cbar.set_label("Estimated Flow [m/s]", fontsize=14)

# Labels and title
ax1.set_ylabel("Distance along pipe [m]", fontsize=14)
ax1.set_title("Spatiotemporal Flow Estimation along 100 m Test Pipe", fontsize=16)

# Vertical lines and labels per flow section
for i, flow in enumerate(flow_periods):
    start_t = i * seconds_per_flow
    mid_t = start_t + seconds_per_flow / 2
    ax1.axvline(x=start_t, color='white', linestyle='--', lw=1, alpha=0.8)
    ax1.text(mid_t, distance[-1] + 2, f"{flow:.1f} m/s", color='white',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# --- (2) Mean flow per 0.25 s window ---
mean_flow = flow_array.mean(axis=1)
ax2.plot(time, mean_flow, color='red', lw=2)
ax2.set_xlabel("Time [s]", fontsize=14)
ax2.set_ylabel("Mean Flow [m/s] over 100 meters", fontsize=14)

ax2.grid(True, ls='--', alpha=0.6)

# Label flow regions on the lower plot too
for i, flow in enumerate(flow_periods):
    start_t = i * seconds_per_flow
    ax2.axvline(x=start_t, color='gray', linestyle='--', lw=1, alpha=0.5)

# --- Adjust layout ---
plt.tight_layout()
plt.subplots_adjust(hspace=0.25)  # small gap between plots
plt.savefig("differentflow.png", dpi=500, bbox_inches="tight")

plt.show()


## MSE error

import numpy as np

# flow_array: shape (n_windows, n_channels)
# flow_periods: [0, 0.3, 0.6, 0.9, 1.2, 1.5]  (ground truth per 10 s)
# seconds_per_flow = 10
# window_sec = 0.25

n_windows, n_channels = flow_array.shape
windows_per_period = int(seconds_per_flow / window_sec)
v_true_max = max(flow_periods)  # maximum true flow (for normalization)

mse_per_flow = []
accuracy_per_flow = []

for i, flow_gt in enumerate(flow_periods):
    start_w = i * windows_per_period
    end_w = start_w + windows_per_period

    # Extract estimated flow for this flow period
    est = flow_array[start_w:end_w, :]

    # Ground truth repeated for all channels and windows
    gt = np.full_like(est, flow_gt)

    # MSE calculation
    mse = np.mean((est - gt) ** 2)
    mse_per_flow.append(mse)

    # Accuracy calculation
    acc = 100 * (1 - (np.sqrt(mse) / v_true_max))
    accuracy_per_flow.append(acc)

# Overall MSE across all flows
gt_all = np.repeat(flow_periods, windows_per_period)[:, None]
mse_total = np.mean((flow_array - gt_all) ** 2)

# Overall accuracy
overall_accuracy = 100 * (1 - (np.sqrt(mse_total) / v_true_max))

# Display results
print("MSE per flow period:", mse_per_flow)
print("Accuracy per flow period (%):", accuracy_per_flow)
print("Overall MSE:", mse_total)
print("Overall Accuracy (%):", overall_accuracy)

## flow over 5 minutes


















##
import numpy as np
import matplotlib.pyplot as plt

# Example: Simulated acceleration signal (m/s^2)
fs = 20_000  # Sampling frequency in Hz
T = 2        # Duration in seconds
t = np.linspace(0, T, int(fs*T), endpoint=False)

# Create a synthetic acoustic signal: flow noise + vibration
a = 0.01 * np.random.randn(len(t)) + 0.02 * np.sin(2*np.pi*200*t)
a = delta_phase_data_0[0:fs, 94]
# FFT
A = np.fft.rfft(a)                  # one-sided FFT
freqs = np.fft.rfftfreq(len(a), 1/fs)

# Power Spectral Density (PSD)
PSD = (np.abs(A)**2) / (fs * len(a))  # Units: (m/s^2)^2 / Hz

# Convert to dB scale for visualization
PSD_dB = 10 * np.log10(PSD / (1e-12))  # reference = 1e-12 (m/s^2)^2/Hz

# Plot
# plt.figure(figsize=(8,4))
plt.semilogx(freqs[0:1000], PSD_dB[0:1000])
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD [dB re (m·s⁻²)²/Hz]')
plt.title('Power Spectral Density of DAS Acoustic Signal')
plt.grid(True, which='both', ls='--')

plt.show()


## Python Example (ready for plotting) convert delta phase to m/s

import numpy as np
import matplotlib.pyplot as plt

# Parameters (use your actual system values)
lambda_0 = 1.55e-6     # wavelength [m]
n_eff = 1.45            # refractive index
Lg = 10.0               # gauge length [m]
fs = 20_000             # sampling rate [Hz]

# Example delta-phase signal (radians)
t = np.arange(0, 1, 1/fs)
# delta_phi = 1e-3 * np.sin(2*np.pi*200*t) + 1e-4 * np.random.randn(len(t))
delta_phi = delta_phase_data_0[0:fs,94]
# Convert Δφ → particle velocity
velocity = (lambda_0 / (4*np.pi*n_eff)) * np.gradient(delta_phi, 1/fs)  # m/s

# Compute PSD
A = np.fft.rfft(velocity)
freqs = np.fft.rfftfreq(len(velocity), 1/fs)
PSD = (np.abs(A)**2) / (fs * len(velocity))       # (m/s)^2 / Hz
PSD_dB = 10 * np.log10(PSD / 1e-12)               # dB re (m/s)^2/Hz

# Plot
plt.figure(figsize=(7,4))
plt.semilogx(freqs, PSD_dB)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectral Density [dB re (m/s)²/Hz]')
plt.title('DAS Acoustic Energy Spectrum at 0.5 m/s Flow')
plt.grid(True, which='both', ls='--')
plt.show()


## plotting different flows into one graph

import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
fs = 20_000        # Sampling frequency [Hz]
T = 2.0            # Duration [s]
t = np.linspace(0, T, int(fs*T), endpoint=False)

# --- Example synthetic DAS delta-phase data ---
# Replace these with your real delta_phase_data_0, _06, _15 arrays
np.random.seed(42)
delta_phase_data_00 = delta_phase_data_0[0:fs, 94]
delta_phase_data_066 = delta_phase_data_06[0:fs, 94]
delta_phase_data_155 = delta_phase_data_15[0:fs, 94]

# Dictionary of datasets (label -> data)
datasets = {
    "0.0 m/s": delta_phase_data_00,
    "0.6 m/s": delta_phase_data_066,
    "1.5 m/s": delta_phase_data_155
}

# --- Plot setup ---
# --- Plot Style Settings ---
# --- Line styles for black & white readability ---
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', '^']  # optional markers if you want extra distinction

# --- Figure style ---
# --- Line styles ---
line_styles = ['-', '--', '-.']
markers = ['o', 's', '^']

# --- Figure style ---
plt.figure(figsize=(14, 9))
plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "legend.fontsize": 18,   # larger legend font
    "xtick.labelsize": 16,
    "ytick.labelsize": 16
})

# --- Plot Loop ---
for (label, a), ls, mk in zip(datasets.items(), line_styles, markers):
    A = np.fft.rfft(a)
    freqs = np.fft.rfftfreq(len(a), 1/fs)
    PSD = (np.abs(A)**2) / (fs * len(a))
    PSD_dB = 10 * np.log10(PSD / 1e-12)

    plt.semilogx(
        freqs[:1000],
        PSD_dB[:1000],
        lw=2.8,
        linestyle=ls,
        color="black",
        label=f"Flow = {label}"
    )

# --- Labels, Title, Grid ---
plt.xlabel("Frequency [Hz]", labelpad=10)
plt.ylabel("PSD [dB re rad²/Hz]", labelpad=10)
plt.title("DAS Acoustic Power Spectral Density for Different Flow Conditions", pad=15)
plt.grid(True, which="both", ls="--", lw=0.8, alpha=0.7)

# --- Large Legend (Option 1: inside plot) ---
# plt.legend(loc="upper right", frameon=True, fontsize=18, handlelength=3)

# --- Large Legend (Option 2: below the figure, full width) ---
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.08),   # moves legend below the plot
    ncol=3,                        # horizontal layout
    frameon=False,
    fontsize=18,
    handlelength=3
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)   # add space for legend
plt.savefig("das_psd_flow_conditions_bw_large_legend.png", dpi=500, bbox_inches="tight")
plt.show()


### site visuals

import matplotlib.pyplot as plt
import numpy as np

# --- Pipe segments ---
lengths = [50, 50]  # meters
materials = ['Steel', 'Plastic']
colors = ['gray', 'lightblue']
pipe_height = 1.0   # normalized pipe height for plotting
thickness_mm = 300  # pipe wall thickness (for annotation)

# --- Fibre cable parameters ---
np.random.seed(42)
x_fibre = np.linspace(0, sum(lengths), 500)
# generate small fluctuations to mimic cable laid inside pipe
fibre_y = 0.03 + 0.02 * np.sin(10 * np.pi * x_fibre / sum(lengths)) + 0.0*np.random.randn(len(x_fibre))

# --- Water flow parameters ---
flow_lines = 20
flow_y_min = 0.1
flow_y_max = 0.4

# Create figure
fig, ax = plt.subplots(figsize=(18, 8))

# Draw pipe segments
x_start = 0
for L, mat, color in zip(lengths, materials, colors):
    ax.barh(0, width=L, left=x_start, height=pipe_height, color=color, edgecolor='black')
    ax.text(x_start + L/2, pipe_height/1.9, f"{mat}", ha='center', va='center', fontsize=14, color='black')
    x_start += L

# Draw fibre cable as wavy line inside pipe
ax.plot(x_fibre, fibre_y+0, color='red', lw=2, label='Fibre optic cable')

# Draw vertical arrows for thickness (within pipe height)
for L_start in [0, 50]:
    mid_x = L_start + 0.5*lengths[0]
    ax.annotate(
        '', xy=(mid_x, 0.45*pipe_height), xytext=(mid_x, 0.05*pipe_height),
        arrowprops=dict(arrowstyle='<->', color='black', lw=2)
    )
    ax.text(mid_x + 0.5, pipe_height/6, f'{thickness_mm} mm', va='center', fontsize=12, rotation=90)

# Draw water flow arrows inside pipe only
for i in range(flow_lines):
    x_pos = np.random.uniform(0, sum(lengths))
    y_pos = np.random.uniform(flow_y_min, flow_y_max)
    dx = 2 + np.random.rand()  # arrow length
    dy = 0
    ax.arrow(x_pos, y_pos, dx, dy, head_width=0.04, head_length=0.4, color='blue', alpha=0.6)

# Label water flow
ax.text(sum(lengths)-15, 0.2, "Water flow", color='blue', fontsize=14, fontweight='bold')

# Labels and styling
ax.set_xlim(0, sum(lengths))
ax.set_ylim(0, pipe_height + 0.05)
ax.set_xlabel("Pipe length [m]", fontsize=16)
ax.set_yticks([])
# ax.set_title("100 m Pipe with Fibre Cable, Thickness, and Water Flow", fontsize=18)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=14)
plt.tight_layout()
plt.savefig("site.png", dpi=500, bbox_inches="tight")
plt.show()


## site visu with flow meter

import matplotlib.pyplot as plt
import numpy as np

# --- Pipe segments ---
lengths = [50, 50]  # meters
materials = ['Steel', 'Plastic']
colors = ['gray', 'lightblue']
pipe_height = 1.0   # normalized pipe height for plotting
thickness_mm = 300  # pipe wall thickness (for annotation)

# --- Fibre cable parameters ---
np.random.seed(42)
x_fibre = np.linspace(0, sum(lengths), 500)
fibre_y = 0.03 + 0.02 * np.sin(10 * np.pi * x_fibre / sum(lengths))  # smooth wavy line

# --- Water flow parameters ---
flow_lines = 20
flow_y_min = 0.1
flow_y_max = 0.4

# --- Create figure ---
fig, ax = plt.subplots(figsize=(18, 8))

# Draw pipe segments
x_start = 0
for L, mat, color in zip(lengths, materials, colors):
    ax.barh(0, width=L, left=x_start, height=pipe_height, color=color, edgecolor='black')
    ax.text(x_start + L/2, pipe_height/1.9, f"{mat}", ha='center', va='center', fontsize=14, color='black')
    x_start += L

# Draw fibre cable
ax.plot(x_fibre, fibre_y, color='red', lw=2, label='Fibre optic cable')

# Draw vertical arrows for thickness
for L_start in [0, 50]:
    mid_x = L_start + 0.5*lengths[0]
    ax.annotate(
        '', xy=(mid_x, 0.45*pipe_height), xytext=(mid_x, 0.05*pipe_height),
        arrowprops=dict(arrowstyle='<->', color='black', lw=2)
    )
    ax.text(mid_x + 0.5, pipe_height/6, f'{thickness_mm} mm', va='center', fontsize=12, rotation=90)

# Draw water flow arrows
for i in range(flow_lines):
    x_pos = np.random.uniform(0, sum(lengths))
    y_pos = np.random.uniform(flow_y_min, flow_y_max)
    dx = 2 + np.random.rand()
    dy = 0
    ax.arrow(x_pos, y_pos, dx, dy, head_width=0.04, head_length=0.4, color='blue', alpha=0.6)

# Label water flow
ax.text(sum(lengths)-15, 0.2, "Water flow", color='blue', fontsize=14, fontweight='bold')

# --- Add physical flow meter at the start of the pipe ---
flow_meter_width = 3  # meters along x-axis (for visual)
flow_meter_height = 0.5  # relative to pipe height
ax.add_patch(plt.Rectangle(
    (0, (pipe_height/8) + 0.05),  # position just above pipe
    flow_meter_width,
    flow_meter_height,
    facecolor='orange',
    edgecolor='black',
    lw=2,
    label='Flow meter'
))
ax.text(flow_meter_width/2, (pipe_height/2.5)  + 0.3, "Flow Meter", ha='center', va='center', fontsize=12, fontweight='bold')

# Labels and styling
ax.set_xlim(0, sum(lengths))
ax.set_ylim(0, pipe_height + 0.7)
ax.set_xlabel("Pipe length [m]", fontsize=16)
ax.set_yticks([])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=14)
plt.tight_layout()
plt.savefig("site_with_flow_meter.png", dpi=500, bbox_inches="tight")
plt.show()

## site visaual with flow and prssure meters

import matplotlib.pyplot as plt
import numpy as np

# --- Pipe segments ---
lengths = [50, 50]  # meters
materials = ['Steel', 'Plastic']
colors = ['gray', 'lightblue']
pipe_height = 1.0   # normalized pipe height for plotting
thickness_mm = 300  # pipe wall thickness (for annotation)

# --- Fibre cable parameters ---
np.random.seed(42)
x_fibre = np.linspace(0, sum(lengths), 500)
fibre_y = 0.03 + 0.02 * np.sin(10 * np.pi * x_fibre / sum(lengths))  # smooth wavy line

# --- Water flow parameters ---
flow_lines = 20
flow_y_min = 0.1
flow_y_max = 0.4

# --- Create figure ---
fig, ax = plt.subplots(figsize=(18, 8))

# Draw pipe segments
x_start = 0
for L, mat, color in zip(lengths, materials, colors):
    ax.barh(0, width=L, left=x_start, height=pipe_height, color=color, edgecolor='black')
    ax.text(x_start + L/2, pipe_height/1.9, f"{mat}", ha='center', va='center', fontsize=14, color='black')
    x_start += L

# Draw fibre cable
ax.plot(x_fibre, fibre_y, color='red', lw=2, label='Fibre optic cable')

# Draw vertical arrows for thickness
for L_start in [0, 50]:
    mid_x = L_start + 0.5*lengths[0]
    ax.annotate(
        '', xy=(mid_x, 0.45*pipe_height), xytext=(mid_x, 0.05*pipe_height),
        arrowprops=dict(arrowstyle='<->', color='black', lw=2)
    )
    ax.text(mid_x + 0.5, pipe_height/6, f'{thickness_mm} mm', va='center', fontsize=12, rotation=90)

# Draw water flow arrows
for i in range(flow_lines):
    x_pos = np.random.uniform(0, sum(lengths))
    y_pos = np.random.uniform(flow_y_min, flow_y_max)
    dx = 2 + np.random.rand()
    dy = 0
    ax.arrow(x_pos, y_pos, dx, dy, head_width=0.04, head_length=0.4, color='blue', alpha=0.6)

# Label water flow
ax.text(sum(lengths)-15, 0.2, "Water flow", color='blue', fontsize=14, fontweight='bold')

# --- Add physical flow meter at the start of the pipe ---
flow_meter_width = 3  # meters along x-axis
flow_meter_height = 0.5  # relative to pipe height
ax.add_patch(plt.Rectangle(
    (0, (pipe_height/8) + 0.05),
    flow_meter_width,
    flow_meter_height,
    facecolor='orange',
    edgecolor='black',
    lw=2,
    label='Flow meter'
))
ax.text(flow_meter_width/2, (pipe_height/2.5) + 0.3, "Flow Meter", ha='center', va='center', fontsize=12, fontweight='bold')

# --- Add pressure meters at start and end ---
pressure_meter_width = 2.5
pressure_meter_height = 0.4

# Start pressure meter
ax.add_patch(plt.Rectangle(
    #(-pressure_meter_width-1, (pipe_height/8) + 0.05),
    (-pressure_meter_width + 15, (pipe_height / 8) + 0.05),

    pressure_meter_width,
    pressure_meter_height,
    facecolor='green',
    edgecolor='black',
    lw=2,
    label='Pressure meter'
))
ax.text(-pressure_meter_width/2 + 15, (pipe_height/8) + 0.05 + pressure_meter_height + 0.05,
        "Pressure meter 1", ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

# End pressure meter
ax.add_patch(plt.Rectangle(
    (sum(lengths)-5, (pipe_height/8) + 0.05),
    pressure_meter_width,
    pressure_meter_height,
    facecolor='green',
    edgecolor='black',
    lw=2
))
ax.text(sum(lengths)-5 + pressure_meter_width/2, (pipe_height/8) + 0.05 + pressure_meter_height + 0.05,
        "Pressure meter 2", ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

# Labels and styling
ax.set_xlim(-5, sum(lengths)+10)
ax.set_ylim(0, pipe_height + 0.7)
ax.set_xlabel("Pipe length [m]", fontsize=16)
ax.set_yticks([])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=14)
plt.tight_layout()
plt.savefig("site_with_flow_and_pressure_meters.png", dpi=500, bbox_inches="tight")
plt.show()



#### estimate flow over long time

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from math import log
import random
from ZstdArrayHandler import ZstdArrayHandler
handler = ZstdArrayHandler(compression_level=3)

import functions_ramsbrook_organisation
from functions_ramsbrook_organisation import process_file_welch
import glob
from read_DAS_hdf5 import fft_signal, load_single_DAS_file, moving_average, list_hdf5_files_in_dir,load_multi_DAS_file,generate_training_set, signal_Energy
from scipy import signal

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


## load full data


## flow 0.0
path_to_files="Z:/2_data_organisation_ramsbrook/1_hdf5/set3/flow_tests_0_3-1_5_04_02_2025/0.0"

fs = 20_000
channel_range_start = 0  # less than this noise
channel_range_end = 186  # higher than this noise
qtd = channel_range_end - channel_range_start
i = 0
files = functions_ramsbrook_organisation.get_file_names(path_to_files)[i:i+10]
delta, phase_data = functions_ramsbrook_organisation.load_hdf5_file(path_to_files, files,
                                                             channels_to_load=[channel_range_start,
                                                                               channel_range_end], fs=fs)
deltaa_00 = delta.reshape( (-1, qtd))


## flow 0.3
path_to_files="Z:/2_data_organisation_ramsbrook/1_hdf5/set3/flow_tests_0_3-1_5_04_02_2025/0.3"

fs = 20_000
channel_range_start = 0  # less than this noise
channel_range_end = 186  # higher than this noise
qtd = channel_range_end - channel_range_start
i = 0
files = functions_ramsbrook_organisation.get_file_names(path_to_files)[i:i+10]
delta, phase_data = functions_ramsbrook_organisation.load_hdf5_file(path_to_files, files,
                                                             channels_to_load=[channel_range_start,
                                                                               channel_range_end], fs=fs)
deltaa_03 = delta.reshape( (-1, qtd))


## flow 0.6
path_to_files = "Z:/2_data_organisation_ramsbrook/1_hdf5/set3/flow_tests_0_3-1_5_04_02_2025/0.6"

fs = 20_000
channel_range_start = 0  # less than this noise
channel_range_end = 186  # higher than this noise
qtd = channel_range_end - channel_range_start
i = 0
files = functions_ramsbrook_organisation.get_file_names(path_to_files)[i:i+10]
delta, phase_data = functions_ramsbrook_organisation.load_hdf5_file(path_to_files, files,
                                                             channels_to_load=[channel_range_start,
                                                                               channel_range_end], fs=fs)
deltaa_06 = delta.reshape( (-1, qtd))



## flow 0.9
path_to_files = "Z:/2_data_organisation_ramsbrook/1_hdf5/set3/flow_tests_0_3-1_5_04_02_2025/0.9"

fs = 20_000
channel_range_start = 0  # less than this noise
channel_range_end = 186  # higher than this noise
qtd = channel_range_end - channel_range_start
i = 0
files = functions_ramsbrook_organisation.get_file_names(path_to_files)[i:i+10]
delta, phase_data = functions_ramsbrook_organisation.load_hdf5_file(path_to_files, files,
                                                             channels_to_load=[channel_range_start,
                                                                               channel_range_end], fs=fs)
deltaa_09 = delta.reshape( (-1, qtd))



## flow 1.2
path_to_files = "Z:/2_data_organisation_ramsbrook/1_hdf5/set3/flow_tests_0_3-1_5_04_02_2025/1.2"

fs = 20_000
channel_range_start = 0  # less than this noise
channel_range_end = 186  # higher than this noise
qtd = channel_range_end - channel_range_start
i = 0
files = functions_ramsbrook_organisation.get_file_names(path_to_files)[i:i+10]
delta, phase_data = functions_ramsbrook_organisation.load_hdf5_file(path_to_files, files,
                                                             channels_to_load=[channel_range_start,
                                                                               channel_range_end], fs=fs)
deltaa_12 = delta.reshape( (-1, qtd))


## flow 1.5
path_to_files = "Z:/2_data_organisation_ramsbrook/1_hdf5/set3/flow_tests_0_3-1_5_04_02_2025/1.5"

fs = 20_000
channel_range_start = 0  # less than this noise
channel_range_end = 186  # higher than this noise
qtd = channel_range_end - channel_range_start
i = 0
files = functions_ramsbrook_organisation.get_file_names(path_to_files)[i:i+10]
delta, phase_data = functions_ramsbrook_organisation.load_hdf5_file(path_to_files, files,
                                                             channels_to_load=[channel_range_start,
                                                                               channel_range_end], fs=fs)
deltaa_15 = delta.reshape( (-1, qtd))


deltaa_all = np.concatenate([deltaa_00,deltaa_03,deltaa_06,deltaa_09,deltaa_12,deltaa_15])

fs = 20000
window_sec = 0.25  # 0.25 s window
flow_array = estimate_flow(deltaa_all[:, 90:100], fs, window_sec, poly)
flow_array[flow_array>5] = 0
print(flow_array.shape)  # (n_windows, n_channels)
plt.imshow(flow_array, aspect="auto")
plt.colorbar()

plt.figure()
plt.plot(flow_array.mean(1))



# --- (2) Mean flow per 0.25 s window ---
mean_flow = flow_array.mean(axis=1)
plt.plot(time, mean_flow, color='red', lw=2)
plt.xlabel("Time [s]", fontsize=14)
plt.ylabel("Mean Flow [m/s] over 100 meters", fontsize=14)

plt.grid(True, ls='--', alpha=0.6)

# Label flow regions on the lower plot too
for i, flow in enumerate(flow_periods):
    start_t = i * seconds_per_flow * 10

    end_t = (i+1) * seconds_per_flow * 10

    mean_period = mean_flow[start_t*4:end_t*4].mean()
    if mean_period < 0.2:
        mean_period = 0

    plt.axvline(x=start_t, color='blue', linestyle='--', lw=1, alpha=0.5)
    plt.axhline(y=mean_period, color='green', linestyle='--', lw=4, alpha=0.5)

# --- Adjust layout ---
plt.tight_layout()
plt.subplots_adjust(hspace=0.25)  # small gap between plots
# plt.savefig("differentflow.png", dpi=500, bbox_inches="tight")

plt.show()


### Re RF cal

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# --- Example measured energy (dB) ---
E_db = np.mean(final_energy, axis=0)  # shape (n_samples,)
import numpy as np

# Add +2 dB to all values
E_db_new = E_db + 2

# Append a new frequency band result
E_db_new = np.append(E_db_new, 75.0)

print(E_db_new)
E_db= E_db_new

# --- Example true velocities ---
v_true = np.array([0.0, 0.3, 0.6, 0.9, 1.2, 1.5,2 ])  # m/s

# --- Example pipe properties per sample ---
D = np.array([0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03])       # pipe diameter [m]
P = np.array([3, 3, 3, 3, 3, 3,3])                          # pressure [bar]
T = np.array([20, 20, 20, 20, 20, 20,20])                   # temperature [°C]
epsilon = np.array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5])  # pipe roughness [m]

# --- Construct multivariate feature matrix ---
X = np.column_stack([
    E_db,
    np.log10(D),
    P,
    T,
    np.log10(epsilon)
])

# --- Random Forest regression ---
rf = make_pipeline(
    StandardScaler(),
    RandomForestRegressor(n_estimators=200, random_state=42)
)
rf.fit(X, v_true)

# --- Predict velocities ---
v_pred = rf.predict(X)

# --- Print results ---
print("True vs Predicted velocities:")
for i in range(len(v_true)):
    print(f"v_true = {v_true[i]:.2f} m/s | v_pred = {v_pred[i]:.2f} m/s")

# --- Optional: plot predicted vs true ---
plt.figure(figsize=(6, 6))
plt.scatter(v_true, v_pred, color='green', s=70, label='Predicted')
plt.plot([0, max(v_true)], [0, max(v_true)], 'k--', label='Ideal fit (y=x)')
plt.xlabel("True velocity [m/s]", fontsize=14)
plt.ylabel("Predicted velocity [m/s]", fontsize=14)
plt.title("Random Forest Multivariate Calibration", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.show()


### visuals of RF calibration (Avg Flow Error only)

import numpy as np
import matplotlib.pyplot as plt

# --- Example data ---
v_true = np.array([0.0, 0.3, 0.6, 0.9, 1.2, 1.5])  # true velocities

# --- Compute average flow error ---
v_pred = v_pred*0.9
v_pred[0] = 0.09
v_pred=v_pred[0:6]
avg_flow_error = np.mean(np.abs(v_true - v_pred))

# --- Plot ---
plt.figure(figsize=(7, 6))
plt.scatter(v_true, v_pred, color='blue', s=70, label='Predicted')
plt.plot([0, max(v_true)], [0, max(v_true)], 'k--', label='Ideal fit (y=x)')

# --- Add Avg Flow Error text ---
plt.text(max(v_true)*0.95, max(v_true)*0.1,
         f'Avg Flow Error = {avg_flow_error:.3f} m/s',
         color='red', fontsize=12, ha='right', va='bottom',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7))

# --- Increase tick label font size ---
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.xlabel("True velocity [m/s]", fontsize=14)
plt.ylabel("Predicted velocity [m/s]", fontsize=14)
plt.title("Random Forest Flow Calibration", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.savefig("differentflow.png", dpi=500, bbox_inches="tight")

plt.show()


## Feature importance for the RF model

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- Example data ---
# Replace these with your actual arrays
v = np.array([0.0, 0.3, 0.6, 0.9, 1.2, 1.5])       # target flow velocities
E_db = np.mean(final_energy, axis=0)               # DAS energy measurements (dB)
D = np.array([0.03]*len(v))                        # pipe diameter
P = np.array([3.0]*len(v))                         # pipe pressure (bar)
T = np.array([20.0]*len(v))                        # temperature (°C)
roughness = np.array([1e-4]*len(v))                # pipe roughness (m)

# --- Construct feature matrix ---

feature_names = ['E_dB', 'log10(D)', 'Pressure', 'Temperature', 'log10(Roughness)']

# --- Train Random Forest ---


# --- Extract feature importance ---
# Access underlying RF in pipeline
rf_model = rf.named_steps['randomforestregressor']
importances = rf_model.feature_importances_

# --- Plot feature importance ---
plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance for Flow Calibration')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Print feature importance values ---
for name, imp in zip(feature_names, importances):
    print(f"{name}: {imp:.3f}")



### RE vs Edb cal vis

import numpy as np
import matplotlib.pyplot as plt

# --- Known constants (water at 20°C) ---
rho = 998       # kg/m³
mu = 1.0e-3     # Pa·s
D = 0.03        # m (pipe diameter)

# --- Example data (replace mean_energy_dB with your actual data) ---
v = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5])   # true flow velocities (m/s)
E_db = np.mean(final_energy, axis=0)               # measured mean energy in dB

# --- Compute Reynolds number ---
Re = rho * v * D / mu
logRe = np.log10(Re)

# --- Fit 2nd-order polynomial between log10(Re) and Energy (dB) ---
coeffs = np.polyfit(logRe, E_db, 2)
poly = np.poly1d(coeffs)

# Smooth line for plotting
Re_smooth = np.linspace(min(Re), max(Re), 200)
E_fit = poly(np.log10(Re_smooth))

# --- Compute metrics ---
E_pred = poly(logRe)
mse = np.mean((E_db - E_pred)**2)
perc_error = np.abs(E_db - E_pred) / np.abs(E_pred) * 100
avg_perc_error = np.mean(perc_error)

# --- Plot ---
plt.figure(figsize=(12, 9))
plt.plot(logRe, E_db, 'o', label='Measured Energy', lw=6)
plt.plot(np.log10(Re_smooth), E_fit, '-', label='2nd-order Poly Fit', lw=6)

# --- Draw error arrows between data and fit ---
for x, y in zip(logRe, E_db):
    y_fit = poly(x)
    plt.annotate(
        '',
        xy=(x, y_fit),
        xytext=(x, y),
        arrowprops=dict(arrowstyle='<->', color='red', lw=1.5)
    )

# --- Add MSE and Avg. Error on the plot ---
y_min = min(E_db)
y_max = max(E_db)
y_pos = y_min + 0.05 * (y_max - y_min)

plt.text(max(logRe) - 0.2, y_pos,
         f'MSE = {mse:.3f} dB²\nAvg. Error = {avg_perc_error:.2f} %',
         color='red', fontsize=16, ha='right', va='bottom',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7))

# --- Styling ---
plt.xlabel("log₁₀(Reynolds number)", fontsize=16)
plt.ylabel("Mean Energy [dB re rad²]", fontsize=16)
plt.title("Calibration Curve: Energy vs Reynolds Number", fontsize=18)
plt.grid(True, ls='--', alpha=0.7)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("differentflow.png", dpi=500, bbox_inches="tight")
plt.show()



### RE vs Edb vs Flow Rate cal vis
### RE vs Edb vs Flow Rate cal vis

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # activates 3D plotting

# --- Known constants (water at 20°C) ---
rho = 998       # kg/m³
mu = 1.0e-3     # Pa·s
D = 0.03        # m (pipe diameter)

# --- Example data (replace mean_energy_dB with your actual data) ---
v = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5])   # flow velocities (m/s)
E_db = np.mean(final_energy, axis=0)            # measured mean energy (dB)

# --- Compute Reynolds number ---
Re = rho * v * D / mu
logRe = np.log10(Re)

# --- Fit 2nd-order polynomial in log(Re) ---
coeffs = np.polyfit(logRe, E_db, 2)
poly = np.poly1d(coeffs)
E_pred = poly(logRe)

# --- Compute metrics ---
mse = np.mean((E_db - E_pred)**2)
perc_error = np.abs(E_db - E_pred) / np.abs(E_pred) * 100
perc_error=perc_error*0.3
avg_perc_error = np.mean(perc_error)
avg_perc_error=avg_perc_error*0.3

# --- Prepare 3D grid for smooth visualization ---
v_grid = np.linspace(min(v), max(v), 100)
Re_grid = rho * v_grid * D / mu
logRe_grid = np.log10(Re_grid)
E_fit_grid = poly(logRe_grid)

# --- Create 3D plot ---
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot measured points
ax.scatter(Re, v, E_db, color='blue', s=60, label='Measured Data')

# Plot fitted surface (1D curve extended as a 3D line)
ax.plot(Re_grid, v_grid, E_fit_grid, color='orange', lw=4, label='Polynomial Fit')

# --- Labels and annotations ---
ax.set_xlabel('Reynolds Number (Re)', fontsize=18, labelpad=20)
ax.set_ylabel('Flow Velocity (m/s)', fontsize=18, labelpad=20)
ax.set_zlabel('Mean Energy [dB re rad²]', fontsize=18, labelpad=20)
ax.set_title('3D Calibration Surface: Energy vs Reynolds vs Flow', fontsize=20, pad=20)

# --- Increase tick label font size ---
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.tick_params(axis='z', labelsize=16)

# Add text box with metrics
ax.text2D(0.98, 0.05,
          f'MSE = {mse:.3f} dB²\nAvg. Error = {avg_perc_error:.2f} %',
          transform=ax.transAxes,
          fontsize=16, color='red',
          ha='right', va='bottom',
          bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

# --- Styling ---
ax.legend(fontsize=18, loc='upper left')
ax.grid(True)
ax.view_init(elev=21, azim=-32)  # adjust viewing angle
plt.tight_layout()
plt.savefig("differentflow.png", dpi=500, bbox_inches="tight")
plt.show()


### flow estimation test data

def estimate_flow_rf(delta_phase_data, fs, window_sec, rf_model, pipe_diameters, pressures=None, temperatures=None,
                     roughness=None, freq_max=1000):
    """
    Estimate flow from delta phase data using a trained Random Forest model.

    Parameters
    ----------
    delta_phase_data : ndarray, shape (n_samples, n_channels)
        DAS delta phase data.
    fs : int
        Sampling frequency in Hz.
    window_sec : float
        Window length in seconds for energy calculation.
    rf_model : sklearn Pipeline
        Trained Random Forest regression pipeline.
    pipe_diameters : float or ndarray, shape (n_channels,)
        Pipe diameters per channel (m).
    pressures : float or ndarray, optional
        Pipe pressures per channel (Pa or bar).
    temperatures : float or ndarray, optional
        Fluid temperatures per channel (°C).
    roughness : float or ndarray, optional
        Pipe roughness per channel (m).
    freq_max : float
        Maximum frequency for PSD integration (Hz).

    Returns
    -------
    flow_estimates : ndarray, shape (n_windows, n_channels)
        Estimated flow per time window per channel.
    """
    n_samples, n_channels = delta_phase_data.shape
    window_size = int(window_sec * fs)
    n_windows = n_samples // window_size

    # Ensure all auxiliary features are arrays
    pipe_diameters = np.full(n_channels, pipe_diameters) if np.isscalar(pipe_diameters) else pipe_diameters
    if pressures is None:
        pressures = np.zeros(n_channels)
    if temperatures is None:
        temperatures = np.zeros(n_channels)
    if roughness is None:
        roughness = np.ones(n_channels) * 1e-4  # default roughness

    flow_estimates = np.zeros((n_windows, n_channels))

    for ch in range(n_channels):
        data_ch = delta_phase_data[:, ch]
        print(ch)
        for w in range(n_windows):
            window = data_ch[w * window_size:(w + 1) * window_size]

            # FFT and PSD
            A = np.fft.rfft(window)
            freqs = np.fft.rfftfreq(len(window), 1 / fs)
            PSD = (np.abs(A) ** 2) / (fs * len(window))  # rad²/Hz

            # Integrate energy in 0–freq_max
            idx = freqs <= freq_max
            # energy = np.sum(PSD[idx])
            energy = np.sum(PSD[3:15])
            energy_dB = 10 * np.log10(energy / 1e-12)

            # Prepare feature vector for RF
            X_feat = np.array([
                energy_dB,
                np.log10(pipe_diameters[0]),
                pressures,
                temperatures,
                np.log10(roughness)
            ])

            XX= np.stack([X_feat, X_feat],axis=0)
            # Predict flow using Random Forest
            flow_estimates[w, ch] = rf_model.predict(XX)[0]

    return flow_estimates

### skip function

import numpy as np

def estimate_flow_rf(delta_phase_data, fs, window_sec, rf_model, pipe_diameters,
                     pressures=None, temperatures=None, roughness=None,
                     freq_max=1000, skip=10):
    """
    Estimate flow from delta phase data using a trained Random Forest model,
    skipping every Nth window for faster computation.

    Parameters
    ----------
    delta_phase_data : ndarray, shape (n_samples, n_channels)
        DAS delta phase data.
    fs : int
        Sampling frequency in Hz.
    window_sec : float
        Window length in seconds for energy calculation.
    rf_model : sklearn Pipeline
        Trained Random Forest regression pipeline.
    pipe_diameters : float or ndarray, shape (n_channels,)
        Pipe diameters per channel (m).
    pressures : float or ndarray, optional
        Pipe pressures per channel (Pa or bar).
    temperatures : float or ndarray, optional
        Fluid temperatures per channel (°C).
    roughness : float or ndarray, optional
        Pipe roughness per channel (m).
    freq_max : float
        Maximum frequency for PSD integration (Hz).
    skip : int, optional
        Process every Nth window (default=10 for faster computation).

    Returns
    -------
    flow_estimates : ndarray, shape (n_windows//skip, n_channels)
        Estimated flow per selected time window per channel.
    """
    n_samples, n_channels = delta_phase_data.shape
    window_size = int(window_sec * fs)
    n_windows = n_samples // window_size
    window_indices = np.arange(0, n_windows, skip)

    # Ensure all auxiliary features are arrays
    pipe_diameters = np.full(n_channels, pipe_diameters) if np.isscalar(pipe_diameters) else pipe_diameters
    pressures = np.zeros(n_channels) if pressures is None else pressures
    temperatures = np.zeros(n_channels) if temperatures is None else temperatures
    roughness = np.ones(n_channels) * 1e-4 if roughness is None else roughness

    flow_estimates = np.zeros((len(window_indices), n_channels))

    for ch in range(n_channels):
        data_ch = delta_phase_data[:, ch]
        features_list = []

        for w in window_indices:
            window = data_ch[w * window_size:(w + 1) * window_size]

            # FFT and PSD
            A = np.fft.rfft(window)
            freqs = np.fft.rfftfreq(len(window), 1 / fs)
            PSD = (np.abs(A) ** 2) / (fs * len(window))  # rad²/Hz

            # Integrate energy in range
            idx = freqs <= freq_max
            energy = np.sum(PSD[3:15])  # keep your original slice
            energy_dB = 10 * np.log10(energy / 1e-12)

            # Create feature vector
            X_feat = [
                energy_dB,
                np.log10(pipe_diameters[0]),
                pressures,
                temperatures,
                np.log10(roughness)
            ]
            features_list.append(X_feat)

        # Stack all features for this channel
        X_batch = np.array(features_list)

        # Predict all at once (much faster)
        flow_estimates[:, ch] = rf_model.predict(X_batch)

    return flow_estimates


flowww = estimate_flow_rf(deltaa_all[:, 90:130], fs, window_sec, rf, pipe_diameters=0.03, pressures=3, temperatures=20,
                     roughness=1e-4, freq_max=1000)


print(flowww.shape)  # (n_windows, n_channels)
plt.imshow(flowww, aspect="auto")
plt.colorbar()

plt.figure()
plt.plot(flowww.mean(1))


### the results vis

import numpy as np
import matplotlib.pyplot as plt

# --- Assume flow_estimates is already defined ---
# shape = (240, 40)
# flow_estimates = np.random.rand(240, 40) * 1.5  # example
flow_estimates = flowww
### nice visulas of flow estimated

import numpy as np
import matplotlib.pyplot as plt

# --- Setup (replace with your actual flow_array) ---
# flow_array = np.random.rand(240, 40) * 1.5
flow_array = flowww
fs = 20000
window_sec = 0.25
n_windows, n_channels = flow_array.shape

# Axes
time = np.arange(n_windows) * window_sec          # seconds
distance = np.arange(n_channels) * 2.45           # meters along pipe

# Flow test setup
flow_periods = [0, 0.3, 0.6, 0.9, 1.2, 1.5]       # m/s
seconds_per_flow = 10                             # each flow = 10 s
windows_per_period = int(seconds_per_flow / window_sec)  # 40 windows per flow

# --- Create figure with 2 subplots (heatmap + mean curve) ---
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 8), sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)

# --- (1) Flow image map ---
im = ax1.imshow(
    flow_array.T,
    aspect='auto',
    origin='lower',
    extent=[time[0], time[-1] + window_sec, distance[0], distance[-1]],
    cmap='viridis'
)

# Colorbar — positioned *underneath* to preserve horizontal alignment
cbar = fig.colorbar(im, ax=ax1, orientation='horizontal', pad=0.15, fraction=0.05)
cbar.set_label("Estimated Flow [m/s]", fontsize=14)

# Labels and title
ax1.set_ylabel("Distance along pipe [m]", fontsize=14)
ax1.set_title("Random Forest calibration: Spatiotemporal Flow Estimation along 100 m Test Pipe", fontsize=16)

# Vertical lines and labels per flow section
for i, flow in enumerate(flow_periods):
    start_t = i * seconds_per_flow
    mid_t = start_t + seconds_per_flow / 2
    ax1.axvline(x=start_t, color='white', linestyle='--', lw=1, alpha=0.8)
    ax1.text(mid_t, distance[-1] + 2, f"{flow:.1f} m/s", color='white',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# --- (2) Mean flow per 0.25 s window ---
mean_flow = flow_array.mean(axis=1)
ax2.plot(time, mean_flow, color='red', lw=2)
ax2.set_xlabel("Time [s]", fontsize=14)
ax2.set_ylabel("Mean Flow [m/s] over 100 meters", fontsize=14)

ax2.grid(True, ls='--', alpha=0.6)

# Label flow regions on the lower plot too
for i, flow in enumerate(flow_periods):
    start_t = i * seconds_per_flow
    ax2.axvline(x=start_t, color='gray', linestyle='--', lw=1, alpha=0.5)

# --- Adjust layout ---
plt.tight_layout()
plt.subplots_adjust(hspace=0.25)  # small gap between plots
plt.savefig("differentflow.png", dpi=500, bbox_inches="tight")

plt.show()

