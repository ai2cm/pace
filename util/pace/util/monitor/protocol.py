from typing import Dict, Protocol

from pace.util.quantity import Quantity


class Monitor(Protocol):
    """
    sympl.Monitor-style object for storing model state dictionaries.
    """

    def store(self, state: dict) -> None:
        """Append the model state dictionary to the zarr store.

        Requires the state contain the same quantities with the same metadata as the
        first time this is called. Dimension order metadata may change between calls
        so long as the set of dimensions is the same. Quantities are stored with
        dimensions [time, rank] followed by the dimensions included in the first
        state snapshot. The one exception is "time" which is stored with dimensions
        [time].
        """
        ...

    def store_constant(self, state: Dict[str, Quantity]) -> None:
        ...

    def cleanup(self):
        ...
