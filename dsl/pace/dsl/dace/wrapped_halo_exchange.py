import dataclasses
from typing import List, Optional

from pace.dsl.dace.orchestration import dace_inhibitor
from pace.util.communicator import CubedSphereCommunicator
from pace.util.halo_updater import HaloUpdater


class WrappedHaloUpdater:
    """Wrapping the original Halo Updater for critical runtime.

    Because DaCe cannot parse the complexity of the HaloUpdater and
    because it cannot pass in Quantity to callback easily, we made a wrapper
    that goes around those two problems, using a .get_attr on a cached state
    to look up the proper quantities.
    """

    def __init__(
        self,
        updater: HaloUpdater,
        state,
        qty_x_names: List[str],
        qty_y_names: List[str] = None,
        comm: Optional[CubedSphereCommunicator] = None,
    ) -> None:
        self._updater = updater
        self._state = state
        self._qtx_x_names = qty_x_names
        self._qtx_y_names = qty_y_names
        self._comm = comm

    @dace_inhibitor
    def start(self):
        if self._qtx_y_names is None:
            if dataclasses.is_dataclass(self._state):
                self._updater.start(
                    [self._state.__getattribute__(x) for x in self._qtx_x_names]
                )
            elif isinstance(self._state, dict):
                self._updater.start([self._state[x] for x in self._qtx_x_names])
            else:
                raise NotImplementedError
        else:
            if dataclasses.is_dataclass(self._state):
                self._updater.start(
                    [self._state.__getattribute__(x) for x in self._qtx_x_names],
                    [self._state.__getattribute__(y) for y in self._qtx_y_names],
                )
            elif isinstance(self._state, dict):
                self._updater.start(
                    [self._state[x] for x in self._qtx_x_names],
                    [self._state[y] for y in self._qtx_y_names],
                )
            else:
                raise NotImplementedError

    @dace_inhibitor
    def wait(self):
        self._updater.wait()

    @dace_inhibitor
    def update(self):
        self.start()
        self.wait()

    @dace_inhibitor
    def interface(self):
        assert len(self._qtx_x_names) == 1
        assert len(self._qtx_y_names) == 1
        self._comm.synchronize_vector_interfaces(
            self._state.__getattribute__(self._qtx_x_names[0]),
            self._state.__getattribute__(self._qtx_y_names[0]),
        )
