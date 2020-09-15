import xarray as xr

__all__ = ["to_dataset"]


def to_dataset(state):
    data_vars = {
        name: value.data_array for name, value in state.items() if name != "time"
    }
    if "time" in state:
        data_vars["time"] = state["time"]
    return xr.Dataset(data_vars=data_vars)
