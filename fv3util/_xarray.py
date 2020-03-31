import xarray as xr

__all__ = ["to_dataset"]


def to_dataset(state):
    return xr.Dataset(
        data_vars={name: value.data_array for name, value in state.items()}
    )
