import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
np.random.seed(42)
n_ch = 100          # number of DAS channels
n_t = 80            # number of time samples
wrap = lambda x: ((x + np.pi) % (2 * np.pi)) - np.pi  # wrap function

# --- Simulation parameters ---
base_rate = np.random.uniform(0.05, 0.2, size=n_ch)  # phase rate before saturation
noise_level = 0.05                                   # small random noise
sat_idx = 50                                         # saturation time index
jump_cycles = np.random.choice([2, 3], size=n_ch)    # 2π or 3π multiple
jump_residual = np.random.uniform(-0.3, 0.3, size=n_ch)  # small non-integer residual

# --- Storage arrays ---
phi_true = np.zeros((n_ch, n_t))
phi_meas = np.zeros((n_ch, n_t))

# --- Generate data ---
for ch in range(n_ch):
    # true continuous phase over time
    phi_true[ch, 0] = 0
    for t in range(1, n_t):
        delta = base_rate[ch] + np.random.randn() * noise_level
        phi_true[ch, t] = phi_true[ch, t - 1] + delta

        # at saturation point, jump by several cycles of 2π
        if t == sat_idx:
            jump_val = 1 * np.pi + jump_residual[ch]
            jump_val = 2 * np.pi * jump_cycles[ch] + jump_residual[ch]

            phi_true[ch, t:] += jump_val  # add to all following samples

    # what DAS measures (wrapped phase)
    phi_meas[ch] = wrap(phi_true[ch])

# --- Plot a few example channels ---
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for i, ax in enumerate(axes):
    ch = i * 30  # show one every ~30 channels
    ax.plot(phi_true[ch], label=f"True Phase (Ch {ch})", linestyle='--', alpha=0.7)
    ax.plot(phi_meas[ch], label=f"Measured (wrapped) Phase", color='C1')
    ax.axvline(sat_idx, color='r', linestyle=':', label="Saturation Point")
    ax.legend(loc='upper left')
    ax.set_ylabel("Phase [rad]")

axes[-1].set_xlabel("Time sample")
plt.suptitle("DAS Phase Simulation with Saturation (Wrap) at t=50")
plt.tight_layout()
plt.show()



plt.plot(phi_meas[10])
plt.plot(phi_true[10])


plt.plot(np.diff(phi_true[10]))