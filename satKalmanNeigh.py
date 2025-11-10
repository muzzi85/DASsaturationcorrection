
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

path_to_files="Z:/1_ds_data_organization_morocco/1_hdf5/1_with_flow/11_06_2024/0000470395_2024-06-11_23.00.48.52288.hdf5"

delta_phase_data, phase_data = load_single_DAS_file(path_to_files)


plt.plot(delta_phase_data[0:2500,450])

### Thomas saturation method

def saturation_metric_accel(diff_phase_sample, limit = 1.4 * np.pi):
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
    return np.count_nonzero(exceed),exceed

ss,s = saturation_metric_accel(delta_phase_data[0:2500,450])


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np

PI = np.pi
TWO_PI = 2*np.pi
PI, TWO_PI = np.pi, 2*np.pi

def wrap(x):
    return (x + PI) % TWO_PI - PI
import numpy as np

PI = np.pi
TWO_PI = 2 * np.pi

def wrap(x):
    return (x + PI) % TWO_PI - PI


def delta_phase_unwrap_v2(delta_phi_wrapped,
                          window_k=4,
                          median_prefilter=False,
                          median_kernel=3,
                          spatial_iter=2):
    """
    Robust unsupervised delta-phase unwrapping starting from delta-phase input.
    - delta_phi_wrapped: (n_ch, n_t) wrapped delta-phase in (-pi, pi]
    - window_k: +/- how many 2pi multiples to test (increase if very large jumps)
    - median_prefilter: if True, apply temporal median filter on wrapped deltas before correction
    - median_kernel: kernel size for median filter (odd)
    - spatial_iter: number of iterative spatial passes to improve results
    Returns:
      unwrapped_phase: (n_ch, n_t) cumulative (absolute) unwrapped phase
      corrected_deltas: (n_ch, n_t) corrected delta-phase values (may exceed +/-pi)
      k_map: (n_ch, n_t) integer 2π multiples applied
    """
    n_ch, n_t = delta_phi_wrapped.shape
    # optional temporal median prefilter (on wrapped values) to reduce impulse noise
    if median_prefilter:
        from scipy.signal import medfilt
        temp = np.zeros_like(delta_phi_wrapped)
        for ch in range(n_ch):
            temp[ch] = medfilt(delta_phi_wrapped[ch], kernel_size=median_kernel)
        work_wrapped = temp
    else:
        work_wrapped = delta_phi_wrapped.copy()

    # outputs
    corrected = np.zeros_like(work_wrapped, dtype=float)
    k_map = np.zeros_like(work_wrapped, dtype=int)

    # Initialize: for t=0 we can't correct temporally; accept wrapped value (or try k that minimizes abs to neighbor median)
    # We'll set corrected[:,0] to wrapped (could also pick best k based on neighbors if desired)
    corrected[:, 0] = work_wrapped[:, 0]
    k_map[:, 0] = 0

    # Main forward pass: use corrected previous deltas as temporal reference,
    # and neighbor corrected deltas (at same time) for spatial adjustment.
    k_candidates = np.arange(-window_k, window_k + 1)

    for t in range(1, n_t):
        # First compute temporal-only correction for each channel
        for ch in range(n_ch):
            d_wr = work_wrapped[ch, t]
            d_prev_corr = corrected[ch, t-1]   # <-- use corrected previous delta (important)
            # choose k that minimizes absolute difference to corrected previous delta
            cand_vals = d_wr + TWO_PI * k_candidates
            errs = np.abs(cand_vals - d_prev_corr)
            k_best = int(k_candidates[np.argmin(errs)])
            d_corr = d_wr + TWO_PI * k_best
            corrected[ch, t] = d_corr
            k_map[ch, t] = k_best

        # Now do spatial correction pass(es)
        for iter_idx in range(spatial_iter):
            # build neighbor-corrected medians for time t
            neigh_med = np.zeros(n_ch)
            for ch in range(n_ch):
                neigh = []
                if ch > 0:
                    neigh.append(corrected[ch-1, t])
                if ch < n_ch - 1:
                    neigh.append(corrected[ch+1, t])
                if neigh:
                    neigh_med[ch] = np.median(neigh)
                else:
                    neigh_med[ch] = corrected[ch, t]

            # adjust each channel if it deviates from neighbor median by > pi (or a threshold)
            for ch in range(n_ch):
                d_corr = corrected[ch, t]
                med = neigh_med[ch]
                # compute integer k_spatial to make d_corr close to med
                k_spatial = int(round((med - d_corr) / TWO_PI))
                if abs((d_corr + k_spatial * TWO_PI) - med) < abs(d_corr - med):
                    # apply spatial correction
                    corrected[ch, t] = d_corr + k_spatial * TWO_PI
                    k_map[ch, t] += k_spatial

    # cumulative sum to absolute unwrapped phase
    unwrapped = np.zeros_like(corrected)
    unwrapped[:, 0] = corrected[:, 0]
    for t in range(1, n_t):
        unwrapped[:, t] = unwrapped[:, t-1] + corrected[:, t]

    return unwrapped, corrected, k_map


# =======================
# Simulation: 100 channels with noise & multi-2π jumps
# =======================


import numpy as np
def unwrap_delta_phase(delta_phi_meas, n_history=3):
    """
    Unwrap a delta phase sequence starting from delta phase.

    Parameters
    ----------
    delta_phi_meas : ndarray, shape (n_t,)
        Measured delta phase (wrapped) for one channel.
    n_history : int
        Number of previous corrected samples to compute gradient.

    Returns
    -------
    delta_phi_corr : ndarray, shape (n_t,)
        Corrected delta phase sequence (unwrapped).
    phi_corr : ndarray, shape (n_t,)
        Integrated phase after unwrapping.
    """
    n_t = len(delta_phi_meas)
    delta_phi_corr = np.zeros_like(delta_phi_meas)
    phi_corr = np.zeros_like(delta_phi_meas)
    phi_diff = np.zeros_like(delta_phi_meas)

    last_wrap_idx = -np.inf  # last index when wrap was corrected
    for t in range(n_t):
        # Start with the raw measured delta
        delta_raw = delta_phi_meas[t]

        # Use previous corrected deltas to compute expected gradient
        if t >= n_history:
            expected = np.mean(delta_phi_corr[t - n_history:t])
        elif t > 0:
            expected = np.mean(delta_phi_corr[:t])
        else:
            expected = delta_raw
        phi_diff[t] = expected
        # Predict potential wrapping: compare delta_raw to expected
        diff = delta_raw - expected

        # Correct if difference is too large (> π or < -π)
        while diff > (np.pi/3):
            #print(t)
            delta_raw -= 2 * np.pi
            diff = delta_raw - expected
            last_wrap_idx = t
        while diff < (-np.pi/3):
            #print(t)
            delta_raw += 2 * np.pi
            diff = delta_raw - expected
            last_wrap_idx = t
        phi_diff[t] = diff

        # Store corrected delta
        delta_phi_corr[t] = delta_raw

        # Integrate to get cumulative phase
        if t == 0:
            phi_corr[t] = delta_phi_corr[t]
        else:
            phi_corr[t] = phi_corr[t - 1] + delta_phi_corr[t]

    return delta_phi_corr, phi_corr, phi_diff


difff = np.diff(phi_meas[10])
delta_phi_corr, phi_corr, phi_diff = unwrap_delta_phase(difff)

### running real moro
ch = 450
for ch in range(450, 455,1):
    difff = delta_phase_data[0:2500, ch]
    delta_phi_corr, phi_corr, phi_diff = unwrap_delta_phase(difff)

    plt.figure()
    plt.plot((difff))
    plt.plot((delta_phi_corr))

    # plt.plot(np.cumsum(delta_phi_corr))
    # plt.plot(np.cumsum(difff))

    plt.xlabel("Time samples")
    plt.ylabel("Corrected phase (cumsum(delta_phase))")
    plt.title(str(ch))

np.random.seed(42)
n_ch = 100
n_t = 80
delta_rates = np.random.uniform(0.05, 0.25, size=n_ch)

delta_phi_wrapped = np.zeros((n_ch, n_t))

for ch in range(n_ch):
    delta_phi = delta_rates[ch] * np.ones(n_t)
    jump_idx = 50
    #jump_idx = np.random.randint(20, 60)

    jump_val = np.random.choice([-2*PI, 2*PI])
    jump_val = 1*PI

    # jump_val = np.random.choice([-6*PI, -4*PI, 4*PI, 6*PI])

    delta_phi[jump_idx:] += jump_val
    noise = np.random.randn(n_t) * 0.1
    delta_phi_noisy = delta_phi + noise
    delta_phi_wrapped[ch] = wrap(delta_phi_noisy)

# Unwrap
unwrapped, corrected, k_map = delta_phase_unwrap_v2(delta_phi_wrapped,
                                                   window_k=4,
                                                   median_prefilter=True,
                                                   median_kernel=3,
                                                   spatial_iter=3)

## plotting

import matplotlib.pyplot as plt

ch = 1
plt.plot(np.cumsum(delta_phi_wrapped[ch]), ':', label='naive cumsum(wrapped)')
plt.plot(unwrapped[ch], '--', label='unwrapped_v2')
plt.legend(); plt.show()

# heatmap of applied k values
plt.imshow(k_map, aspect='auto', cmap='bwr', interpolation='nearest')
plt.colorbar(label='k (2π multiples applied)')
plt.xlabel('time'); plt.ylabel('channel')
plt.title('k_map (2π multiples applied)')
plt.show()



# =======================
# Animation: circle visualization
# =======================
n_vis_ch = 20
theta = wrap(unwrapped_phase[:n_vis_ch])
correction_flags = corrections[:n_vis_ch]

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.set_title('Delta-phase DAS Unwrapping on Circle with Noise and Multi-2π Correction')

# circle outline
circle = plt.Circle((0,0), 1, color='black', fill=False)
ax.add_artist(circle)

dots = [ax.plot([], [], 'o')[0] for _ in range(n_vis_ch)]
lines = [ax.plot([], [], 'r-', alpha=0.2)[0] for _ in range(n_vis_ch-1)]
trail_len = 5
trail_positions = [[] for _ in range(n_vis_ch)]

def init():
    for dot in dots:
        dot.set_data([], [])
    for line in lines:
        line.set_data([], [])
    return dots + lines

def animate(i):
    for ch, dot in enumerate(dots):
        x = np.cos(theta[ch, i])
        y = np.sin(theta[ch, i])

        # highlight if multi-2pi correction applied
        if abs(correction_flags[ch, i]) > 0:
            dot.set_color('orange')
            dot.set_markersize(8)
        else:
            dot.set_color('blue')
            dot.set_markersize(5)

        dot.set_data(x, y)

        # update trail
        trail_positions[ch].append((x, y))
        if len(trail_positions[ch]) > trail_len:
            trail_positions[ch].pop(0)

        # neighbor influence line
        if ch < n_vis_ch-1:
            x1, y1 = x, y
            x2, y2 = np.cos(theta[ch+1, i]), np.sin(theta[ch+1, i])
            lines[ch].set_data([x1, x2], [y1, y2])

    return dots + lines

anim = FuncAnimation(fig, animate, frames=n_t, init_func=init, interval=150, blit=True)
plt.show()


###

