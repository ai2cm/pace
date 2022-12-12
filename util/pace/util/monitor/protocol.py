from typing import Dict, Protocol

from pace.util.quantity import Quantity


class Monitor(Protocol):
    """
    sympl.Monitor-style object for storing model state dictionaries.
    """

    def store(self, state: dict) -> None:
        """Append the model state dictionary to the stored data."""
        ...

    def store_constant(self, state: Dict[str, Quantity]) -> None:
        ...

    def cleanup(self):
        ...
