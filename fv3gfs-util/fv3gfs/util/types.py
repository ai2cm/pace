from typing import Iterable, Tuple, TypeVar

from typing_extensions import Protocol


Array = TypeVar("Array")


class Allocator(Protocol):
    def __call__(self, shape: Iterable[int], dtype: type) -> Array:
        pass


class NumpyModule(Protocol):

    empty: Allocator
    zeros: Allocator
    ones: Allocator

    def rot90(self, m: Array, k: int = 1, axes: Tuple[int, int] = (0, 1)):
        ...


class AsyncRequest(Protocol):
    """Define the result of an over-the-network capable communication API"""

    def wait(self):
        """Block the current thread waiting for the request to be completed"""
        ...
