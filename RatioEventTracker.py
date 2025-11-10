import csv
import json
import uuid
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
#import numpy.typing as npt


# logger = structlog.get_logger(f"{__name__}.ratio_detector")


class EventTracker:
    def __init__(
        self,
        sta_win: int,
        lta_win: int,
        threshold: float,
        alarm_window_seconds: int,
        alarm_event_count: int,
        duration_threshold_seconds: int,
    ):
        if lta_win <= sta_win:
            raise ValueError("LTA window must be larger than STA window.")

        self.sta_win = sta_win
        self.lta_win = lta_win
        self.threshold = threshold

        # Alarm parameters
        self.alarm_window = timedelta(seconds=alarm_window_seconds)
        self.alarm_event_count = alarm_event_count
        self.duration_threshold = timedelta(seconds=duration_threshold_seconds)

        # --- Internal State ---
        self.data_buffer = deque(maxlen=lta_win)  # type: ignore
        self.event_timestamps = deque()  # type: ignore
        self._is_currently_in_event = False
        self._current_event_start_time: Optional[datetime] = None

    def reset(self):
        """Wipes the event memory and resets the state."""
        #logger.info("ALARM MEMORY WIPED. Tracking restarted.")
        self.event_timestamps.clear()
        self._is_currently_in_event = False
        self._current_event_start_time = None

    def process_chunk(self, data_chunk: np.float32, ts: datetime, channel: int) -> dict:
        self.data_buffer.append(data_chunk)
        sta_avg, lta_avg, ratio = None, None, None
        event_detected, alarm_triggered = False, False

        if len(self.data_buffer) < self.lta_win:
            return {
                "timestamp": ts,
                "sta": sta_avg,
                "lta": lta_avg,
                "ratio": ratio,
                "event": event_detected,
                "alarm": alarm_triggered,
            }

        lta_avg = sum(self.data_buffer) / self.lta_win
        sta_samples = list(self.data_buffer)[-self.sta_win :]
        sta_avg = sum(sta_samples) / self.sta_win
        ratio = sta_avg / lta_avg if lta_avg > 0 else 100.0

        # --- Event and Duration Alarm Logic ---
        if ratio > self.threshold:
            if not self._is_currently_in_event:
                # Event START
                self._is_currently_in_event = True
                self.event_timestamps.append(ts)
                event_detected = True
                self._current_event_start_time = ts
                #logger.info(f"Event detected at {channel}. (Ratio: {ratio:.2f}).")
            else:
                # Event CONTINUES - check duration
                if self._current_event_start_time and (
                    ts - self._current_event_start_time > self.duration_threshold
                ):
                    #logger.info(
                        #f"!!! DURATION ALARM!!! \n Event lasted longer than {self.duration_threshold.seconds} seconds."
                  #  )
                    self.reset()
                    alarm_triggered = True
        elif ratio < self.threshold and self._is_currently_in_event:
            # Event END
            self._is_currently_in_event = False
            self._current_event_start_time = None

        # --- Event Count Alarm Logic (only check if another alarm hasn't just fired) ---
        if not alarm_triggered:
            while (
                self.event_timestamps
                and self.event_timestamps[0] < ts - self.alarm_window
            ):
                self.event_timestamps.popleft()
            if len(self.event_timestamps) >= self.alarm_event_count:
               # logger.info(
               #     f"!!! EVENT COUNT ALARM !!! \n {len(self.event_timestamps)} events occurred in the last {self.alarm_window.seconds} seconds"
               # )

                self.reset()
                alarm_triggered = True

        return {
            "timestamp": ts,
            "sta": sta_avg,
            "lta": lta_avg,
            "ratio": ratio,
            "event": event_detected,
            "alarm": alarm_triggered,
        }


def append_list_to_csv(filename, data_list):
    with open(filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(data_list)


def log_status(chunk, timestamp, results, channel, args, alarm=False):
    # add timestamp flavours for rotation (date_str) and for file-writes (datetime_str)
    date_str = timestamp.date().strftime("%Y-%m-%d")
    datetime_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    if alarm:
        log_data = [
            chunk,
            channel,
            datetime_str,
            results["sta"],
            results["lta"],
            results["ratio"],
            results["event"],
            results["alarm"],
        ]
        alarm_output_filename = args.log_folder.joinpath("alarms.csv")
        append_list_to_csv(alarm_output_filename, log_data)

    else:
        log_data = [
            chunk,
            datetime_str,
            results["sta"],
            results["lta"],
            results["ratio"],
            results["event"],
            results["alarm"],
        ]
        output_filename = args.log_folder.joinpath(
            f"event_log_channel{channel}_{date_str}.csv"
        )
        append_list_to_csv(output_filename, log_data)


def calculate_nominations(
    trackers,
    arr,
    channels_to_process,
    chunk: int,
    timestamp,
    args,
):
    # reshape to put channels as axis=0 so we can loop over them
    # the order of channels will be the same as order informed in the config
    # additionally, it will squeeze unnecessary dimension of time
    # (given that we always get 1 sec)
    # and sum the frequencies of interest
    arr = (
        arr.transpose(1, 0, 2)[:, :, args.freq_range[0] : args.freq_range[1]]
        .squeeze(1)
        .sum(1)
    )

    for index, (channels_index, channel_data) in enumerate(
        zip(channels_to_process, arr)
    ):
        results = trackers[index].process_chunk(channel_data, timestamp, channels_index)
        log_status(chunk, timestamp, results, channels_index, args, alarm=False)
        if results["alarm"]:
            log_status(chunk, timestamp, results, channels_index, args, alarm=True)
            nomination = {
                "fiberChannel": channels_index,
                "severity": 2,
                "metadata": {
                    "algorithm": "RatioDetector",
                },
                "alarmTime": timestamp.isoformat(),
                "alarmId": str(uuid.uuid4()),
                "statusBit": None,
                "messageType": "SYSTEM_ALARM",
                "classification": "DRILLING",
            }

            # if args.send_to_observer:
            #     with redis.Redis.from_url(str(args.redis_uri)) as client:
            #         client.publish("system:events", json.dumps(nomination))
