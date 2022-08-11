import abc
import logging
from typing import Mapping, Optional

import numpy as np

import pace.util


logger = logging.getLogger(__name__)


class StencilComparison(abc.ABC):
    @abc.abstractmethod
    def compare(self, function_name: str, **data):
        """
        Args:
            function_name: name of function being tested
            data: data to be compared
        """
        ...


class ParallelModelComparison(StencilComparison):
    """
    Class used to compare two copies of the model running under one global MPI context.

    Comparison for the whole tile is done on the first rank of each tile,
    allowing the number of ranks to differ between the two models but
    increasing memory required.

    Can only compare Quantity values, since this requires communication
    and grid staggering information.
    """

    def __init__(
        self,
        global_comm: pace.util.Comm,
        tile_communicator: pace.util.TileCommunicator,
        pair_rank: Optional[int] = None,
    ):
        """
        Args:
            global_comm: mpi4py-style communicator which allows communicating
                across the global MPI context
            tile_communicator: communication class for the tile of the local rank
            pair_ranks: if this rank is the first rank on a tile, the global rank of
                the first process on the identical tile for the other model copy,
                otherwise give None
        """
        self._global_comm = global_comm
        self._rank = self._global_comm.Get_rank()
        self._tile_communicator = tile_communicator
        self._pair_rank = pair_rank

    def compare(self, function_name: str, **data):
        """
        Args:
            function_name: name of function being tested
            data: data to be compared
        """
        data = {
            name: value
            for (name, value) in data.items()
            if isinstance(value, pace.util.Quantity)
        }
        tile_data: Mapping[
            str, pace.util.Quantity
        ] = self._tile_communicator.gather_state(data)
        if self._pair_rank is not None and len(tile_data) > 0:
            tile_compute_data = {
                name: value.view[:] for name, value in tile_data.items()
            }
            logger.debug(
                "comparing %s variables with rank %s",
                len(tile_compute_data),
                self._pair_rank,
            )
            differences = compare_ranks(
                self._global_comm, tile_compute_data, pair_rank=self._pair_rank
            )
            if len(differences) > 0:
                raise ValueError(
                    f"rank {self._global_comm.Get_rank()} has differences "
                    f"{differences} "
                    f"before or after calling {function_name}"
                )


class TwinModelComparison(StencilComparison):
    """
    Class used to compare two copies of the model running under one global MPI context.

    This class only supports cases where the models have the same number of ranks.
    """

    def __init__(
        self,
        global_comm: pace.util.Comm,
    ):
        """
        Args:
            global_comm: mpi4py-style communicator which allows communicating
                across the global MPI context
            tile_communicator: communication class for the tile of the local rank
            pair_ranks: mapping whose keys are all global ranks corresponding
                to the first rank of a tile, and values are the first rank of
                the identical tile on the other copy of the model
        """
        self._global_comm = global_comm

    def compare(self, function_name: str, **data):
        """
        Args:
            function_name: name of function being tested
            data: data to be compared
        """
        pair_rank = get_pair_rank(
            self._global_comm.Get_rank(), self._global_comm.Get_size()
        )
        differences = compare_ranks(self._global_comm, data, pair_rank=pair_rank)
        if len(differences) > 0:
            raise ValueError(
                f"rank {self._global_comm.Get_rank()} has differences {differences} "
                f"before or after calling {function_name}"
            )


def get_pair_rank(rank: int, size: int):
    dycore_ranks = size // 2
    if rank < dycore_ranks:
        return rank + dycore_ranks
    else:
        return rank - dycore_ranks


def compare_ranks(comm: pace.util.Comm, data, pair_rank: int) -> Mapping[str, int]:
    differences = {}
    for name, maybe_array in sorted(data.items(), key=lambda x: x[0]):
        if isinstance(maybe_array, pace.util.Quantity):
            maybe_array = maybe_array.data
        if hasattr(maybe_array, "data") and isinstance(maybe_array.data, np.ndarray):
            array = maybe_array.data
            other = comm.sendrecv(array, pair_rank)
            arr_diffs = np.sum(np.logical_and(~np.isnan(array), array != other))
            if arr_diffs > 0:
                differences[name] = arr_diffs
    return differences
