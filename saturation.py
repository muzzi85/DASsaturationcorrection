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
    return n2


def detect_and_correct_wrapping(phases_wrapped, return_wrap_points=False):
    """
    Detect and correct phase wrapping (saturation) in DAS-like systems.

    Parameters
    ----------
    phases_wrapped : array-like
        Sequence of measured wrapped phase values (in radians), each within [-π, +π].
    return_wrap_points : bool, optional
        If True, return indices where wrapping corrections occurred.

    Returns
    -------
    phases_unwrapped : np.ndarray
        Continuous (unwrapped) phase array.
    wrap_indices : list of int, optional
        Indices where a wrap (±2π correction) was applied.
    """
    phases_wrapped = np.asarray(phases_wrapped)
    phases_unwrapped = np.zeros_like(phases_wrapped)
    phases_unwrapped[0] = phases_wrapped[0]

    wrap_indices = []

    for i in range(1, len(phases_wrapped)):
        # Compute phase difference
        delta = phases_wrapped[i] - phases_wrapped[i - 1]

        # Wrap delta into (-π, π]
        delta_corrected = (delta + np.pi) % (2 * np.pi) - np.pi

        # Detect if correction was applied
        if not np.isclose(delta, delta_corrected):
            wrap_indices.append(i)

        # Reconstruct unwrapped phase
        phases_unwrapped[i] = phases_unwrapped[i - 1] + delta_corrected

    if return_wrap_points:
        return phases_unwrapped, wrap_indices
    else:
        return phases_unwrapped


## load leak data from Moroccow
import glob
path_to_files = f'D:/drilling 13-06-2024/0000483419_2024-06-13_11.41.49.08836.hdf5'#120000850-144500850_Feb-02'#f'Z:/1_ds_data_organization_morocco/1_hdf5/1_with_flow/11_06'#120000850-144500850_Feb-02


phase_data = load_single_DAS_file(path_to_files)

ch = 3070
ch_485 = phase_data[0:1000, ch]
ch_485_corected = detect_and_correct_wrapping(ch_485)

plt.plot(ch_485, label="Original")
plt.plot(ch_485_corected, label="Corrected")
plt.xlabel("Samples")
plt.ylabel("Phase pi to -pi")
plt.legend()  # <-- shows the legend box
plt.title("Moro channel " + str(ch))
plt.show()


plt.figure()
plt.plot(phase_data.mean(0))
plt.xlabel("DAS channels")
plt.ylabel("Phase  (mean over 10 seconds)")