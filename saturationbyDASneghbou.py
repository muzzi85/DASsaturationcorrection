

import h5py
def load_single_DAS_file(file_name):
    hf = h5py.File(file_name, 'r')
    n1 = hf.get('DAS')
    n2 = np.array(n1)
    n2 = n2 * (np.pi / 2 ** 15)
    #print(f'[HDF5 Processing] Integrate')
    n22 = np.cumsum(n2,axis=0)
    return n2, n22


## simulate das
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# ------------------------------------------------------------------
# USER PARAMETERS
# ------------------------------------------------------------------
λ  = 1.55e-6          # laser wavelength   (m)
n  = 1.468            # fibre refr. index
ξ  = 0.78             # photo-elastic coeff
L  = 10.0             # gauge length       (m)
dz = 1.0              # channel spacing    (m)

# safety margin for declaring a temporal wrap
WRAP_THRESH = np.pi - 0.2

# ------------------------------------------------------------------
# CORE ROUTINE
# ------------------------------------------------------------------
def correct_phase_wrap(m):
    """
    m : ndarray, complex128, shape (n_ch, n_t)
        Raw IQ data from DAS interrogator or the original phase
    Returns
    -------
    strain : ndarray, float64, same shape (µε)
    """
    n_ch, n_t = m.shape

    # 1. temporal gradient (channel-wise)
    ## gives you the delta phase of areas that is saturated
    ## this cancels out the amplitude and isolates the phase difference between two neighboring time stamps
    dφ_t = np.angle(m[:, 1:] * np.conj(m[:, :-1]))   # (n_ch, n_t-1)
    #dφ_t = m   # (n_ch, n_t-1)

    ### below is an example of measuring angle difference in the exp space or (sine cosine or reality vs imaginary )
    # two consecutive IQ samples from one channel
    # z1 = 1 + 1j  # 45°
    # z2 = -1 + 1j  # 135°
    #
    # # temporal phase step
    # dphi = np.angle(z1 * np.conj(z2))
    # print(dphi * 180 / np.pi)  # → -90.0  (45° - 135° = -90°)
    #
    ### below is an example of measuring angle difference in the exp space or (sine cosine or reality vs imaginary )
    # Array example (same as in the DAS code)
    # m = np.array([[1 + 0j, 0 + 1j, -1 + 0j],  # ch 0
    #               [1 + 1j, -1 + 1j, -1 - 1j]])  # ch 1
    #
    # dphi_t = np.angle(m[:, 1:] * np.conj(m[:, :-1]))
    # print(dphi_t * 180 / np.pi)
    # [[ 90. -90.]   # ch0:  90° step, then -90° step
    #  [  0.  90.]]  # ch1:   0° step, then  90° step


    # 2. flag saturated channels per time sample
    ## highlight the saturated channels
    bad = np.zeros_like(m, dtype=bool)
    bad[:, 1:] = np.abs(dφ_t) > WRAP_THRESH

    # 3. spatial gradient (never wraps under proper sampling)
    ## highligh angle difference between two phase points, between two neighbour channels
    ## this cancels out the amplitude and isolates the phase difference between two neighboring channels
    dφ_s = np.angle(m[1:, :] * np.conj(m[:-1, :]))   # (n_ch-1, n_t)

    # 4. integrate spatially to rebuild absolute phase
    φ = np.zeros_like(m, dtype=float)          # rebuilt phase (rad)
    # seed first good channel with measured wrapped phase
    φ[0, :] = np.angle(m[0, :])

    # forward integration
    for k in range(1, n_ch):
        φ[k, :] = φ[k-1, :] + dφ_s[k-1, :]

    # 5. optional: overwrite only bad places (keeps good untouched)
    #    comment out these two lines if you want the pure spatial rebuild
    good = ~bad
    φ[good] = np.angle(m[good])

    # 6. convert to micro-strain
    scale = λ / (4*np.pi*n*ξ*L) * 1e6   # 1e6 → µε
    strain = φ * scale
    return φ, bad

# dummy data: 128 ch, 10 000 t
m = np.random.normal(0, 1, (128, 10000))  # replace with real data
mm = m.view(complex)  # replace with real data

strain_µε, bad = correct_phase_wrap(mm)

## load ap sensing data

## load leak data from Moroccow
import glob
path_to_files = f'Z:/1_ds_data_organization_morocco/1_hdf5/1_with_flow/11_06/0000470390_2024-06-11_22.59.58.52288.hdf5'#120000850-144500850_Feb-02'#f'Z:/1_ds_data_organization_morocco/1_hdf5/1_with_flow/11_06'#120000850-144500850_Feb-02

path_to_files = f"Z:/1_ds_data_organization_morocco/1_hdf5/2_without_flow/26_11_2024/0001402821_2024-11-26_23.11.43.20665.hdf5"


path_to_files = "Y:/3_data_organisation_jordan/1_hdf5/1_high_flow/040000791-050000791/0002181914_2025-04-14_04.00.38.32812.hdf5"
delta_phase_data, phase_data = load_single_DAS_file(path_to_files)
phase_data=phase_data.T
delta_phase_data=delta_phase_data.T



# ------------------------------------------------------------------
# CORE ROUTINE
# ------------------------------------------------------------------
import numpy as np

# ----------- USER CONSTANTS --------------------------------------
λ  = 1.55e-6          # m
n  = 1.468
ξ  = 0.78
L  = 10.0             # m
WRAP_THRESH = np.pi - 0.2   # saturation threshold
# ------------------------------------------------------------------

def correct_from_dphi_t(dphi_t):
    """
    dphi_t : ndarray, float64, shape (n_ch, n_t-1)
        Delta-phase between consecutive time samples (already wrapped)
    Returns
    -------
    phi_corr : ndarray, float64, shape (n_ch, n_t)
        Reconstructed / corrected absolute phase (rad)
    bad      : ndarray, bool,      shape (n_ch, n_t)
        Mask of samples declared saturated
    """
    n_ch, n_t1 = dphi_t.shape
    n_t = n_t1 + 1

    # 1. temporal unwrap per channel → proxy absolute phase
    phi = np.zeros((n_ch, n_t))
    phi[:, 1:] = np.cumsum(dphi_t, axis=1)          # integrate time
    # optional: 1-D unwrap along time (removes 2π jumps)
    phi = np.unwrap(phi, axis=1)

    # 2. flag saturated samples (large jump in dphi_t)
    bad = np.zeros((n_ch, n_t), dtype=bool)
    bad[:, 1:] = np.abs(dphi_t) > WRAP_THRESH

    # 3. build proxy complex signal
    m_proxy = np.exp(1j * phi)          # unit amplitude is enough

    # 4. spatial gradient from proxy
    dphi_s = np.angle(m_proxy[1:, :] * np.conj(m_proxy[:-1, :]))

    # 5. spatial integration to rebuild phase
    phi_corr = np.zeros_like(phi)
    phi_corr[0, :] = phi[0, :]          # seed with unwrapped time integral
    for k in range(1, n_ch):
        phi_corr[k, :] = phi_corr[k-1, :] + dphi_s[k-1, :]

    # 6. paste back good regions (optional)
    good = ~bad
    phi_corr[good] = phi[good]

    return phi_corr, bad


row = delta_phase_data[0:, 0:10000]

r = np.ascontiguousarray(row)
r = r.view(complex)
WRAP_THRESH=np.pi/50
strain_µε, bad = correct_from_dphi_t(row)

plt.imshow((bad.T),aspect="auto",vmax=1)
plt.title("Saturation over 2pi*0.7 - during no flow time of Morocow")
plt.ylabel("samples")
plt.xlabel("DAS channels")

## plotting
plt.plot(bad[1,:100])
plt.plot(dφ_t[1,:100])
plt.plot(m[1,:100])
plt.axhline(y =WRAP_THRESH,color='r')


## thomas saturation model


def saturation_metric_accel(diff_phase_sample, limit= 1.4 * np.pi):
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
    ## ploting
    # plt.figure()
    # plt.plot(accel_phase)
    # plt.title("Acceleration phase")
    # plt.axhline(y=limit, color="r")
    # plt.xlabel("samples or 1/4 sec")
    #
    # plt.figure()
    # plt.plot(diff_phase_sample)
    # plt.title("Delta phase")
    # plt.xlabel("samples or 1/4 sec")
    #
    # plt.figure()
    # plt.plot(exceed)
    # plt.title("saturaton areas")
    # plt.xlabel("samples or 1/4 sec")

    return np.count_nonzero(exceed)

thomas_sat = np.zeros((delta_phase_data.shape[0],delta_phase_data.shape[1]))
limit= np.pi/5
for ch in range(thomas_sat.shape[0]):
    if ch % 1000 == 0:
        print(ch)
    count = 0
    for timee in range(0,10000, 100):
        # print(timee)
        thomas_satt = saturation_metric_accel(delta_phase_data[ch, timee:timee+5000],limit)
        #print(thomas_satt)
        thomas_sat[ch, count] = thomas_satt
        count+=1


thomas_sat_f = np.copy(thomas_sat[:, 0:100])
# thomas_sat_f[thomas_sat_f>1] = 100
plt.imshow((thomas_sat_f.T),aspect="auto",vmin=0, vmax=1)
# plt.title("Saturation over 2pi*0.7 - during no flow time of Morocow")
plt.title("Saturation over 1.4 * pi - during flow time of Jordan")
plt.ylabel("samples")
plt.xlabel("DAS channels")



### correct from delta phase

def correct_from_dphi(dφ_t, dφ_s, λ, n, ξ, L, WRAP_THRESH=np.pi):
    """
    Correct wrapped DAS delta-phase measurements using spatial info.
    dφ_t : ndarray, shape (n_ch, n_t-1)  temporal phase differences (rad)
    dφ_s : ndarray, shape (n_ch-1, n_t)  spatial phase differences (rad)
    """
    n_ch, n_t1 = dφ_t.shape
    n_t = n_t1 + 1

    # 1. Identify saturated temporal jumps
    bad = np.zeros((n_ch, n_t), dtype=bool)
    bad[:, 1:] = np.abs(dφ_t) > WRAP_THRESH

    # 2. Integrate spatially to rebuild absolute phase (seed = 0)
    φ = np.zeros((n_ch, n_t))
    for k in range(1, n_ch):
        φ[k, :] = φ[k-1, :] + dφ_s[k-1, :]

    # 3. Apply temporal integration to propagate phase evolution
    for k in range(n_ch):
        for i in range(1, n_t):
            if bad[k, i]:
                # replace bad temporal step with spatially consistent estimate
                φ[k, i] = φ[k, i-1]  # keep previous phase (simple example)
            else:
                φ[k, i] = φ[k, i-1] + dφ_t[k, i-1]

    # 4. Convert to microstrain
    scale = λ / (4*np.pi*n*ξ*L) * 1e6
    strain = φ * scale
    return strain, bad



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------------
# 1. Simulation setup
# -------------------------
n_time, n_chan = 15, 4
x = np.zeros((n_time, n_chan))
for k in range(n_chan):
    x[:, k] = np.linspace(0, 3, n_time)

# Introduce saturation / large jumps
x[5, 1] += 5 * np.pi  # channel 1 >4π jump
x[7, 2] += 3 * np.pi  # channel 2 >2π jump
x[6, 3] -= 4 * np.pi  # channel 3 negative jump

# Wrap to [-pi, pi]
x_wrapped = (x + np.pi) % (2 * np.pi) - np.pi


# -------------------------
# 2. Robust neighbor-based unwrapping
# -------------------------
def robust_unwrap_with_neighbors(x, R=np.pi, max_multiple=5):
    n_time, n_chan = x.shape
    z = np.exp(1j * np.pi * x / R)
    delta_phi = np.angle(z[1:] - z[:-1])
    delta_phi_corrected = np.copy(delta_phi)
    corrections = np.zeros_like(delta_phi)  # store 2π multiples applied

    for n in range(delta_phi.shape[0]):
        for k in range(n_chan):
            neighbors = []
            if k > 0:
                neighbors.append(delta_phi[n, k - 1])
            if k < n_chan - 1:
                neighbors.append(delta_phi[n, k + 1])
            if neighbors:
                neighbors = np.array(neighbors)
                possible_m = np.arange(-max_multiple, max_multiple + 1)
                errors = [np.sum(np.abs(delta_phi[n, k] + 2 * np.pi * m - neighbors)) for m in possible_m]
                best_m = possible_m[np.argmin(errors)]
                delta_phi_corrected[n, k] = delta_phi[n, k] + 2 * np.pi * best_m
                corrections[n, k] = best_m  # record correction

    # Integrate over time
    x_unwrapped = np.zeros_like(x)
    x_unwrapped[0, :] = x[0, :]
    for n in range(1, n_time):
        x_unwrapped[n, :] = x_unwrapped[n - 1, :] + (R / np.pi) * delta_phi_corrected[n - 1, :]

    return x_unwrapped, corrections


x_unwrapped, corrections = robust_unwrap_with_neighbors(x_wrapped)

# -------------------------
# 3. Animation setup
# -------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
ax_circle = axes[0, 0]
ax_lines_wrapped = axes[0, 1]
ax_lines_unwrapped = axes[1, 0]
axes[1, 1].axis('off')  # empty subplot

# Unit circle
circle = plt.Circle((0, 0), 1, color='gray', fill=False, lw=2)
ax_circle.add_artist(circle)
points_wrapped, = ax_circle.plot([], [], 'ro', label='Wrapped')
points_unwrapped, = ax_circle.plot([], [], 'bo', label='Unwrapped')
ax_circle.set_xlim([-1.2, 1.2])
ax_circle.set_ylim([-1.2, 1.2])
ax_circle.set_aspect('equal')
ax_circle.grid(True)
ax_circle.legend()
ax_circle.set_title("Phase on Unit Circle")

# Line plots
lines_wrapped, lines_unwrapped = [], []
time_steps = np.arange(n_time)
for k in range(n_chan):
    lw, = ax_lines_wrapped.plot([], [], 'o-', label=f'Ch {k}')
    lines_wrapped.append(lw)
    lu, = ax_lines_unwrapped.plot([], [], 'o-', label=f'Ch {k}')
    lines_unwrapped.append(lu)

for ax, title in zip([ax_lines_wrapped, ax_lines_unwrapped], ["Wrapped Phase", "Unwrapped Phase"]):
    ax.set_xlim(0, n_time - 1)
    ax.set_ylim(np.min(x_unwrapped) - 1, np.max(x_unwrapped) + 1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Phase (rad)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()


# -------------------------
# 4. Animation functions
# -------------------------
def init():
    points_wrapped.set_data([], [])
    points_unwrapped.set_data([], [])
    for lw, lu in zip(lines_wrapped, lines_unwrapped):
        lw.set_data([], [])
        lu.set_data([], [])
        lw.set_color('blue')
        lu.set_color('blue')
    return points_wrapped, points_unwrapped, *lines_wrapped, *lines_unwrapped


def animate(i):
    # Circle points
    xw = np.cos(x_wrapped[i])
    yw = np.sin(x_wrapped[i])
    points_wrapped.set_data(xw, yw)

    xu = np.cos(x_unwrapped[i] % (2 * np.pi))
    yu = np.sin(x_unwrapped[i] % (2 * np.pi))
    points_unwrapped.set_data(xu, yu)

    # Line plots with correction highlighting
    for k in range(n_chan):
        lines_wrapped[k].set_data(time_steps[:i + 1], x_wrapped[:i + 1, k])
        lines_unwrapped[k].set_data(time_steps[:i + 1], x_unwrapped[:i + 1, k])
        # Highlight if correction applied at this step
        if i < corrections.shape[0] and corrections[i, k] != 0:
            lines_unwrapped[k].set_color('red')
        else:
            lines_unwrapped[k].set_color('blue')

    return points_wrapped, points_unwrapped, *lines_wrapped, *lines_unwrapped


# -------------------------
# 5. Create animation
# -------------------------
ani = FuncAnimation(fig, animate, frames=n_time, init_func=init,
                    blit=True, interval=1500, repeat=False)  # slower

plt.tight_layout()
plt.show(block=True)  # keep window open in PyCharm
