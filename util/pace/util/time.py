import datetime

import cftime
import numpy as np


# Calendar constant values copied from time_manager in FMS
THIRTY_DAY_MONTHS = 1
JULIAN = 2
GREGORIAN = 3
NOLEAP = 4
FMS_TO_CFTIME_TYPE = {
    THIRTY_DAY_MONTHS: cftime.Datetime360Day,
    JULIAN: cftime.DatetimeJulian,
    GREGORIAN: cftime.DatetimeGregorian,  # Not a valid calendar in FV3GFS
    NOLEAP: cftime.DatetimeNoLeap,
}


def datetime64_to_datetime(dt64: np.datetime64) -> datetime.datetime:
    utc_start = np.datetime64(0, "s")
    timestamp = (dt64 - utc_start) / np.timedelta64(1, "s")
    return datetime.datetime.utcfromtimestamp(timestamp)
