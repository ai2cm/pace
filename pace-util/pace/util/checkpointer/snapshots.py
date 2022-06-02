import collections

import numpy as np
import xarray as xr

from .base import Checkpointer


def make_dims(savepoint_dim, label, data_list):
    data = np.concatenate([array[None, :] for array in data_list], axis=0)
    dims = [savepoint_dim] + [f"{label}_dim{i}" for i in range(len(data.shape[1:]))]
    return dims, data


class Snapshots:
    def __init__(self):
        self._savepoints = collections.defaultdict(list)
        self._python_data = collections.defaultdict(list)

    def compare(self, savepoint_name: str, variable_name: str, python_data):
        self._savepoints[variable_name].append(savepoint_name)
        self._python_data[variable_name].append(python_data)

    @property
    def dataset(self) -> xr.Dataset:
        data_vars = {}
        for variable_name, savepoint_list in self._savepoints.items():
            savepoint_dim = f"sp_{variable_name}"
            data_vars[f"{variable_name}_savepoints"] = ([savepoint_dim], savepoint_list)
            data_vars[f"{variable_name}"] = make_dims(
                savepoint_dim, variable_name, self._python_data[variable_name]
            )
        return xr.Dataset(data_vars=data_vars)


class SnapshotCheckpointer(Checkpointer):
    def __init__(self, rank: int):
        self._rank = rank
        self._snapshots = Snapshots()

    def __call__(self, savepoint_name, **kwargs):
        for name, value in kwargs.items():
            array_data = np.copy(value.data)
            self._snapshots.compare(savepoint_name, name, array_data)

    @property
    def dataset(self):
        return self._snapshots.dataset

    def cleanup(self):
        self.dataset.to_netcdf(f"comparison_rank{self._rank}.nc")
