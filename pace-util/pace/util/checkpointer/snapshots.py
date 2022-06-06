import collections

import numpy as np

from pace.util._optional_imports import xarray as xr

from .base import Checkpointer


def make_dims(savepoint_dim, label, data_list):
    """
    Helper which defines dimension names for an xarray variable.

    Used to ensure no dimensions have the same name but different sizes
    when defining xarray datasets.
    """
    data = np.concatenate([array[None, :] for array in data_list], axis=0)
    dims = [savepoint_dim] + [f"{label}_dim{i}" for i in range(len(data.shape[1:]))]
    return dims, data


class _Snapshots:
    def __init__(self):
        self._savepoints = collections.defaultdict(list)
        self._arrays = collections.defaultdict(list)

    def store(self, savepoint_name: str, variable_name: str, python_data):
        self._savepoints[variable_name].append(savepoint_name)
        self._arrays[variable_name].append(python_data)

    @property
    def dataset(self) -> "xr.Dataset":
        data_vars = {}
        for variable_name, savepoint_list in self._savepoints.items():
            savepoint_dim = f"sp_{variable_name}"
            data_vars[f"{variable_name}_savepoints"] = ([savepoint_dim], savepoint_list)
            data_vars[f"{variable_name}"] = make_dims(
                savepoint_dim, variable_name, self._arrays[variable_name]
            )
        if xr is None:
            raise ModuleNotFoundError(
                "xarray must be installed to use Snapshots.dataset"
            )
        else:
            return xr.Dataset(data_vars=data_vars)


class SnapshotCheckpointer(Checkpointer):
    """
    Checkpointer which can be used to save datasets showing the evolution
    of variables between checkpointer calls.
    """

    def __init__(self, rank: int):
        if xr is None:
            raise ModuleNotFoundError(
                "xarray must be installed to use SnapshotCheckpointer"
            )
        self._rank = rank
        self._snapshots = _Snapshots()

    def __call__(self, savepoint_name, **kwargs):
        for name, value in kwargs.items():
            array_data = np.copy(value.data)
            self._snapshots.store(savepoint_name, name, array_data)

    @property
    def dataset(self) -> "xr.Dataset":
        return self._snapshots.dataset

    def cleanup(self):
        self.dataset.to_netcdf(f"comparison_rank{self._rank}.nc")
