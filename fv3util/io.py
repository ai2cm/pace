import xarray as xr
from .time import datetime64_to_datetime


def write_state(state, filename):
    if 'time' not in state:
        raise ValueError('state must include a value for "time"')
    ds = xr.Dataset(data_vars=state)
    ds.to_netcdf(filename)


def read_state(filename):
    out_dict = {}
    ds = xr.open_dataset(filename)
    for name, value in ds.data_vars.items():
        if name == 'time':
            out_dict[name] = datetime64_to_datetime(value)
        else:
            out_dict[name] = value
    return out_dict
