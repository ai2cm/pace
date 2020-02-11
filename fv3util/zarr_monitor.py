from typing import Union
import logging
import zarr
import numpy as np

logger = logging.getLogger("fv3util")


class DummyComm:
    
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def bcast(self, value, root=0):
        assert root == 0, "DummyComm should only be used on a single core, so root should only ever be 0"
        return value

    def barrier(self):
        return


class ZarrMonitor:
    """sympl.Monitor-style object for storing model states in a Zarr store."""

    def __init__(
            self,
            store: Union[str, zarr.storage.MutableMapping],
            mode: str = "w",
            mpi_comm=DummyComm()):
        """Create a ZarrMonitor.

        Args:
            store: Zarr store in which to store data
            mode: mode to use to open the store. Options are as in zarr.open_group.
            mpi_comm: mpi4py comm object to use for communications. By default, will
                use a dummy comm object that works in single-core mode.
        """
        self._group = zarr.open_group(store, mode=mode)
        self._comm = mpi_comm
        self._rank = mpi_comm.Get_rank()
        self._total_ranks = mpi_comm.Get_size()
        self._prepend_shape = [1, self._total_ranks]
        self._PREPEND_CHUNKS = [1, 1]
        self._PREPEND_DIMS = ["time", "rank"]
        self._writers = None

    def _init_writers(self, state):
        self._writers = {
            key: _ZarrVariableWriter(self._comm, self._group, name=key)
            for key in set(state.keys()).difference(['time'])
        }
        self._writers['time'] = _ZarrTimeWriter(self._comm, self._group, name='time')

    def _check_writers(self, state):
        extra_names = set(state.keys()).difference(self._writers.keys())
        if len(extra_names) != 0:
            raise ValueError(
                f"provided state has keys {extra_names} "
                "that were not present in earlier states"
            )
        missing_names = set(self._writers.keys()).difference(state.keys())
        if len(missing_names) != 0:
            raise ValueError(
                f"provided state is missing keys {missing_names} "
                "that were present in earlier states"
            )

    def _ensure_writers_are_consistent(self, state):
        if self._writers is None:
            self._init_writers(state)
        else:
            self._check_writers(state)

    def store(self, state):
        """Append the model state to the zarr store.

        Requires the state contain the same quantities with the same metadata as the
        first time this is called. Quantities are stored with dimensions [time, rank]
        followed by the dimensions included in any one state snapshot. The one exception
        is "time" which is stored with dimensions [time].
        """
        self._ensure_writers_are_consistent(state)
        for name, array in state.items():
            self._writers[name].append(array)


class _ZarrVariableWriter:

    def __init__(self, comm, group, name):
        self.i_time = 0
        self.comm = comm
        self.group = group
        self.name = name
        self.array = None
        self._prepend_shape = (1, self.size)
        self._prepend_chunks = (1, 1)
        self._PREPEND_DIMS = ("time", "rank")

    @property
    def rank(self):
        return self.comm.Get_rank()

    @property
    def size(self):
        return self.comm.Get_size()

    def _init_zarr(self, array):
        if self.rank == 0:
            self._init_zarr_root(array)
        self.sync_array()

    def _init_zarr_root(self, array):
        shape = self._prepend_shape + array.shape
        chunks = self._prepend_chunks + array.shape
        self.array = self.group.create_dataset(
            self.name, shape=shape, dtype=array.dtype, chunks=chunks
        )

    def set_dims(self, dims):
        if self.rank == 0:
            self.array.attrs["_ARRAY_DIMENSIONS"] = dims

    def sync_array(self):
        self.array = self.comm.bcast(self.array, root=0)

    def append(self, array):
        # can't just use array.append because we only want to
        # extend the dimension once, from the master rank
        if self.array is None:
            self._init_zarr(array)
            self.array.attrs.update(array.attrs)
            self.set_dims(self._PREPEND_DIMS + array.dims)

        if self.i_time >= self.array.shape[0] and self.rank == 0:
            new_shape = (self.i_time + 1, self.size) + self.array.shape[2:]
            self.array.resize(*new_shape)
            self._ensure_compatible_attrs(array)
        self.sync_array()
        self.array[self.i_time, self.rank, ...] = np.asarray(array)
        self.i_time += 1

    def _ensure_compatible_attrs(self, new_array):
        new_attrs = {'_ARRAY_DIMENSIONS': list(self._PREPEND_DIMS + new_array.dims)}
        new_attrs.update(new_array.attrs)
        if dict(self.array.attrs) != new_attrs:
            raise ValueError(
                f"value for {self.name} with attrs {new_attrs} "
                f"does not match previously stored attrs {dict(self.array.attrs)}"
            )


class _ZarrTimeWriter(_ZarrVariableWriter):

    def __init__(self, *args, **kwargs):
        super(_ZarrTimeWriter, self).__init__(*args, **kwargs)
        self._prepend_shape = (1,)
        self._prepend_chunks = (1,)

    def append(self, time):
        array = np.array(np.datetime64(time))
        if self.array is None:
            self._init_zarr(array)
            self.set_dims(["time"])

        if self.i_time >= self.array.shape[0] and self.rank == 0:
            new_shape = (self.i_time + 1,)
            self.array.resize(*new_shape)
        self.sync_array()
        if self.rank == 0:
            self.array[self.i_time, ...] = np.asarray(array)
        self.i_time += 1
        self.comm.barrier()
