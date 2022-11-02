import functools
from typing import Iterable, TypeVar

import numpy as np
from typing_extensions import Protocol


Array = TypeVar("Array")


class Allocator(Protocol):
    def __call__(self, shape: Iterable[int], dtype: type) -> Array:
        pass


class NumpyModule(Protocol):

    empty: Allocator
    zeros: Allocator
    ones: Allocator

    @functools.wraps(np.rot90)
    def rot90(self, *args, **kwargs):
        ...

    @functools.wraps(np.sum)
    def sum(self, *args, **kwargs):
        ...

    @functools.wraps(np.log)
    def log(self, *args, **kwargs):
        ...

    @functools.wraps(np.sin)
    def sin(self, *args, **kwargs):
        ...

    @functools.wraps(np.asarray)
    def asarray(self, *args, **kwargs):
        ...


class AsyncRequest(Protocol):
    """Define the result of an over-the-network capable communication API"""

    def wait(self):
        """Block the current thread waiting for the request to be completed"""
        ...
