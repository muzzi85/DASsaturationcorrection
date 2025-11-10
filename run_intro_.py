import numpy as np
import RatioEventTracker
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import csv
import pandas as pd
from RatioEventTracker import append_list_to_csv

data = np.load(
    "C:\Muzzi work\introusion\muzzi_code\coil_tempering_fft100_jordan_c7.npy"
)  # data is in 10:510 Hz ]# 840-940 ( c6 to c7 ) - c7 : 912

plt.pcolormesh(data.sum(2))
plt.axvline(x=58)
plt.xlabel("Channel")
plt.ylabel("Time")

##

tracker = RatioEventTracker.EventTracker(
    sta_win=1,
    lta_win=100,
    threshold=8,
    alarm_window_seconds=60,
    alarm_event_count=3,
    duration_threshold_seconds=10,
)

channel_to_run = 58  # channel that we will be tracking

data_in = data[:, channel_to_run, :].sum(
    1
)  # just selectiong specific channel and summing energy


output_filename = "test.csv"

for second in data_in:
    process_timestamp = datetime.now(timezone.utc)

    results = tracker.process_chunk(second, process_timestamp, channel_to_run)
    log_data = [
        channel_to_run,
        process_timestamp,
        results["sta"],
        results["lta"],
        results["ratio"],
        results["event"],
        results["alarm"],
    ]
    append_list_to_csv(output_filename, log_data)

