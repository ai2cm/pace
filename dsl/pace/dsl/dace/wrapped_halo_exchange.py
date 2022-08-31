from typing import List, Optional, Union

import numpy as np

from pace.dsl.dace.orchestration import dace_inhibitor
from pace.util.halo_updater import HaloUpdater, VectorInterfaceHaloUpdater


class WrappedHaloUpdater:
    """Wrapping the original Halo Updater for critical runtime.

    Because DaCe cannot parse the complexity of the HaloUpdater and
    because it cannot pass in Quantity to callback easily, we made a wrapper
    that goes around those two problems, using a .get_attr on a cached state
    to look up the proper quantities.
    """

    def __init__(
        self,
        updater: Union[HaloUpdater, VectorInterfaceHaloUpdater],
    ) -> None:
        self._updater = updater

    @dace_inhibitor
    def start(
        self, arrays_x: List[np.ndarray], arrays_y: Optional[List[np.ndarray]] = None
    ):
        assert isinstance(self._updater, HaloUpdater)
        self._updater.start(arrays_x, arrays_y)

    @dace_inhibitor
    def wait(self):
        self._updater.wait()

    @dace_inhibitor
    def update(
        self, arrays_x: List[np.ndarray], arrays_y: Optional[List[np.ndarray]] = None
    ):
        self.start(arrays_x, arrays_y)
        self.wait()

    @dace_inhibitor
    def interface(self, arrays_x: np.ndarray, arrays_y: np.ndarray):
        assert isinstance(self._updater, VectorInterfaceHaloUpdater)
        request = self._updater.start_synchronize_vector_interfaces(arrays_x, arrays_y)
        request.wait()
