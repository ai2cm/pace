import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import fsspec
import numpy as np

from pace.util.communicator import Communicator

from .. import _xarray as xr
from ..filesystem import get_fs
from ..quantity import Quantity
from .convert import to_numpy


logger = logging.getLogger(__name__)


class _TimeChunkedVariable:
    def __init__(self, initial: Quantity, time_chunk_size: int):
        self._data = np.zeros(
            (time_chunk_size, *initial.extent), dtype=initial.data.dtype
        )
        self._data[0, ...] = to_numpy(initial.view[:])
        self._dims = initial.dims
        self._units = initial.units
        self._i_time = 1

    def append(self, quantity: Quantity):
        self._data[self._i_time, ...] = to_numpy(quantity.transpose(self._dims).view[:])
        self._i_time += 1

    @property
    def data(self) -> Quantity:
        return Quantity(
            data=self._data[: self._i_time, ...],
            dims=("time",) + tuple(self._dims),
            units=self._units,
        )


class _ChunkedNetCDFWriter:

    FILENAME_FORMAT = "state_{chunk:04d}.nc"

    def __init__(self, path: str, fs: fsspec.AbstractFileSystem, time_chunk_size: int):
        self._path = path
        self._fs = fs
        self._time_chunk_size = time_chunk_size
        self._i_time = 0
        self._chunked: Optional[Dict[str, _TimeChunkedVariable]] = None
        self._times: List[Any] = []
        self._time_units: Optional[str] = None

    def append(self, state):
        logger.debug("appending at time %d", self._i_time)
        state = {**state}  # copy so we don't mutate the input
        time = state.pop("time", None)
        if self._chunked is None:
            self._chunked = {
                name: _TimeChunkedVariable(quantity, self._time_chunk_size)
                for name, quantity in state.items()
            }
        else:
            for name, quantity in state.items():
                self._chunked[name].append(quantity)
        self._times.append(time)
        if (self._i_time + 1) % self._time_chunk_size == 0:
            logger.debug("flushing on append at time %d", self._i_time)
            self.flush()
        self._i_time += 1

    def flush(self):
        if self._chunked is None:
            pass
        else:
            data_vars = {"time": (["time"], self._times)}
            for name, chunked in self._chunked.items():
                data_vars[name] = xr.DataArray(
                    to_numpy(chunked.data.view[:]),
                    dims=chunked.data.dims,
                    attrs=chunked.data.attrs,
                )
            ds = xr.Dataset(data_vars=data_vars)
            chunk_index = self._i_time // self._time_chunk_size
            chunk_path = str(
                Path(self._path)
                / _ChunkedNetCDFWriter.FILENAME_FORMAT.format(chunk=chunk_index)
            )
            with self._fs.open(chunk_path, "wb") as f:
                ds.to_netcdf(f)

        self._chunked = None
        self._times.clear()


class NetCDFMonitor:
    """
    sympl.Monitor-style object for storing model state dictionaries netCDF files.
    """

    _CONSTANT_FILENAME = "constants.nc"

    def __init__(
        self,
        path: str,
        communicator: Communicator,
        time_chunk_size: int = 1,
    ):
        """Create a NetCDFMonitor.

        Args:
            path: directory in which to store data
            communicator: provides global communication to gather state
            filename_format: a string format for the filename. Must contain a
                "chunk" field which will be replaced with the chunk number.
            time_chunk_size: number of times
        """
        self._path = path
        self._fs = get_fs(path)
        self._communicator = communicator
        self._time_chunk_size = time_chunk_size
        self.__writer: Optional[_ChunkedNetCDFWriter] = None
        self._expected_vars: Optional[Set[str]] = None

    @property
    def _writer(self):
        if self.__writer is None:
            self.__writer = _ChunkedNetCDFWriter(
                path=self._path,
                fs=self._fs,
                time_chunk_size=self._time_chunk_size,
            )
        return self.__writer

    def store(self, state: dict) -> None:
        """Append the model state dictionary to the netcdf files.

        Will only write to disk when a full time chunk has been accumulated,
        or when .cleanup() is called.

        Requires the state contain the same quantities with the same metadata as the
        first time this is called. Dimension order metadata may change between calls
        so long as the set of dimensions is the same. Quantities are stored with
        dimensions [time, tile] followed by the dimensions included in the first
        state snapshot. The one exception is "time" which is stored with dimensions
        [time].
        """
        if self._expected_vars is None:
            self._expected_vars = set(state.keys())
        elif self._expected_vars != set(state.keys()):
            raise ValueError(
                "state keys must be the same each time store is called, "
                "got {} but previously got {}".format(
                    set(state.keys()), self._expected_vars
                )
            )
        state = self._communicator.gather_state(state)
        if state is not None:  # we are on root rank
            self._writer.append(state)

    def store_constant(self, state: Dict[str, Quantity]) -> None:
        state = self._communicator.gather_state(state)
        if state is not None:  # we are on root rank
            constants_filename = str(
                Path(self._path) / NetCDFMonitor._CONSTANT_FILENAME
            )
            if self._fs.exists(constants_filename):
                with self._fs.open(constants_filename, "rb") as f:
                    ds = xr.open_dataset(f)
                    ds = ds.load()
                for name, quantity in state.items():
                    ds[name] = xr.DataArray(
                        to_numpy(quantity.view[:]),
                        dims=quantity.dims,
                        attrs=quantity.attrs,
                    )
            else:
                ds = xr.Dataset(
                    data_vars={
                        name: xr.DataArray(
                            to_numpy(value.view[:]),
                            dims=value.dims,
                            attrs=value.attrs,
                        )
                        for name, value in state.items()
                    }
                )
            with self._fs.open(constants_filename, "wb") as f:
                ds.to_netcdf(f)

    def cleanup(self):
        self._writer.flush()
