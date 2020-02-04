import numpy as np
import datetime


def datetime64_to_datetime(dt64):
    utc_start = np.datetime64(0, 's')
    timestamp = (dt64 - utc_start) / np.timedelta64(1, 's')
    return datetime.utcfromtimestamp(timestamp)
