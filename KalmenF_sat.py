import numpy as np

class PhaseKalman:
    def __init__(self, dt,
                 q_phi=1e-4, q_phi_dot=1e-2,
                 r_meas=1e-2,
                 phi0=0.0, phi_dot0=0.0,
                 P0=None):
        """
        Simple 2-state Kalman filter for phase tracking with integer-wrap correction.

        dt: sampling interval
        q_phi, q_phi_dot: process noise variances for phi and phi_dot
        r_meas: measurement noise variance (on phase, radians^2)
        phi0, phi_dot0: initial state guess
        P0: initial covariance (2x2) or None to set sensible default
        """
        self.dt = dt
        # state: [phi, phi_dot]
        self.x = np.array([phi0, phi_dot0], dtype=float)

        # State transition
        self.F = np.array([[1.0, dt],
                           [0.0, 1.0]])

        # Process noise covariance Q
        self.Q = np.array([[q_phi, 0.0],
                           [0.0, q_phi_dot]])

        # Measurement matrix H = [1, 0]
        self.H = np.array([[1.0, 0.0]])

        # Measurement noise
        self.R = np.array([[r_meas]])

        # Covariance
        if P0 is None:
            # large initial uncertainty in phi_dot, moderate in phi
            self.P = np.array([[1.0, 0.0],
                               [0.0, 1.0]])
        else:
            self.P = P0

    @staticmethod
    def wrap_phase(x):
        """Wrap phase into (-pi, pi]"""
        return (x + np.pi) % (2*np.pi) - np.pi

    def predict(self):
        # x = F x
        self.x = self.F.dot(self.x)
        # P = F P F^T + Q
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        return self.x.copy()

    def update(self, z_wrapped):
        """
        z_wrapped: measured wrapped phase in radians (in -pi..pi]
        returns: updated state
        """
        # predicted phase (scalar)
        phi_pred = float(self.x[0])

        # Determine integer k to map z_wrapped to the branch closest to phi_pred:
        k = np.round((phi_pred - z_wrapped) / (2*np.pi))
        z_unwrapped = z_wrapped + 2*np.pi*k

        # Innovation
        y = z_unwrapped - (self.H.dot(self.x))[0]

        # Innovation covariance S = H P H^T + R
        S = float(self.H.dot(self.P).dot(self.H.T) + self.R)
        # Kalman gain
        K = (self.P.dot(self.H.T) / S).reshape(2)

        # State update
        self.x = self.x + K * y

        # Covariance update: P = (I - K H) P
        KH = np.outer(K, self.H[0])
        self.P = (np.eye(2) - KH).dot(self.P)

        return self.x.copy()

    def step(self, z_wrapped):
        """
        One predict+update step using wrapped measurement z_wrapped.
        """
        self.predict()
        return self.update(z_wrapped)




import numpy as np
## testing on real data


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
    return n22


## load leak data from Moroccow
import glob
path_to_files = f'Z:/3_data_organisation_jordan/1_hdf5/1_high_flow/14_04_night_1/0002180419_2025-04-14_00.00.08.32812.hdf5'#120000850-144500850_Feb-02'#f'Z:/1_ds_data_organization_morocco/1_hdf5/1_with_flow/11_06'#120000850-144500850_Feb-02

phase_data = load_single_DAS_file(path_to_files)


phi_dot_true = 10.0 * 2 * np.pi  # rad/s (10 Hz cycles per second in 2pi units)
phi_true = phase_data[0:1000, 512]
# Create KF
# simulation parameters
dt = 0.001
T = 10.0
t = np.arange(0, T, dt)

# Wrapped measurements with noise
meas_noise_std = 0.05  # rad
z_wrapped = ((phi_true + np.pi) % (2 * np.pi)) - np.pi
z_wrapped_noisy = z_wrapped + np.random.randn(len(z_wrapped)) * meas_noise_std
z_wrapped_noisy = np.array([((z + np.pi) % (2 * np.pi)) - np.pi for z in z_wrapped_noisy])



ch = 520
z_wrapped_noisy = phase_data[0:10000, ch]

z_wrapped = ((z_wrapped_noisy + np.pi) % (2*np.pi)) - np.pi
plt.plot(z_wrapped)
# Create KF
kf = PhaseKalman(dt=dt,
                 q_phi=1e-6, q_phi_dot=1e-1,
                 r_meas=meas_noise_std ** 2,
                 phi0=z_wrapped_noisy[0], phi_dot0=phi_dot_true * 0.9)

est = np.zeros((len(t), 2))
for i in range(len(t)):
    est[i] = kf.step(z_wrapped[i])

# Plot results
plt.figure(figsize=(10, 6))
#plt.plot(t, phi_true / (2 * np.pi), label='True phase (cycles)')
plt.plot(t, np.cumsum(z_wrapped_noisy) / (2 * np.pi), label='Cumsum of wrapped meas (bad)')
plt.figure()
plt.plot(t, est[:, 0] / (2 * np.pi), label='KF estimate (unwrapped)')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Phase [cycles]')
plt.title('Kalman-based phase unwrapping via prediction + integer shift')
plt.grid(True)
plt.show()

# -------------------------------
# Example simulation and usage:
# -------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # simulation parameters
    dt = 0.01
    T = 6.0
    t = np.arange(0, T, dt)

    # true phase: a ramp (fast) + sinusoid to create multiple wraps
    phi_dot_true = 10.0 * 2*np.pi   # rad/s (10 Hz cycles per second in 2pi units)
    phi_true = 0.5 * (phi_dot_true) * t + 4.0*np.sin(2*np.pi*1.5*t)  # linear growth + oscillation

    # Wrapped measurements with noise
    meas_noise_std = 0.05  # rad
    z_wrapped = ((phi_true + np.pi) % (2*np.pi)) - np.pi
    z_wrapped_noisy = z_wrapped + np.random.randn(len(z_wrapped)) * meas_noise_std
    z_wrapped_noisy = np.array([((z+np.pi)%(2*np.pi))-np.pi for z in z_wrapped_noisy])

    # Create KF
    kf = PhaseKalman(dt=dt,
                     q_phi=1e-6, q_phi_dot=1e-1,
                     r_meas=meas_noise_std**2,
                     phi0=z_wrapped_noisy[0], phi_dot0=phi_dot_true*0.9)

    est = np.zeros((len(t), 2))
    for i in range(len(t)):
        est[i] = kf.step(z_wrapped_noisy[i])

    # Plot results
    plt.figure(figsize=(10,6))
    plt.plot(t, phi_true/(2*np.pi), label='True phase (cycles)')
    plt.plot(t, np.cumsum(z_wrapped_noisy)/(2*np.pi), label='Cumsum of wrapped meas (bad)')
    plt.plot(t, est[:,0]/(2*np.pi), label='KF estimate (unwrapped)')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Phase [cycles]')
    plt.title('Kalman-based phase unwrapping via prediction + integer shift')
    plt.grid(True)
    plt.show()
