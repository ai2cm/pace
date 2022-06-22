import dataclasses
import os
from typing import Dict, Protocol, Union

import numpy as np
import xarray as xr

from .grid import Grid


def dataset_to_dict(ds: xr.Dataset) -> Dict[str, Union[np.ndarray, float, int]]:
    return {
        name: _process_if_scalar(array.values) for name, array in ds.data_vars.items()
    }


def _process_if_scalar(value: np.ndarray) -> Union[np.ndarray, float, int]:
    if len(value.shape) == 0:
        return value.item()
    else:
        return value


class Translate(Protocol):
    def collect_input_data(self, ds: xr.Dataset) -> dict:
        ...

    def compute(self, data: dict):
        ...


@dataclasses.dataclass
class SavepointCase:
    """
    Represents a savepoint with data on one rank.
    """

    savepoint_name: str
    data_dir: str
    rank: int
    i_call: int
    testobj: Translate
    grid: Grid

    def __str__(self):
        return f"{self.savepoint_name}-rank={self.rank}-call={self.i_call}"

    @property
    def ds_in(self) -> xr.Dataset:
        return (
            xr.open_dataset(os.path.join(self.data_dir, f"{self.savepoint_name}-In.nc"))
            .isel(rank=self.rank)
            .isel(savepoint=self.i_call)
        )

    @property
    def ds_out(self) -> xr.Dataset:
        return (
            xr.open_dataset(
                os.path.join(self.data_dir, f"{self.savepoint_name}-Out.nc")
            )
            .isel(rank=self.rank)
            .isel(savepoint=self.i_call)
        )
