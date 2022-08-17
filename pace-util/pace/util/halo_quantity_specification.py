from dataclasses import dataclass
from .types import NumpyModule
from typing import Tuple, Any
from .quantity import Quantity


@dataclass
class QuantityHaloSpec:
    """Describe the memory to be exchanged.

    Specification needs to cover all aspect of the memory layout for
    borth scalar, vector and interface fields, for their halo to be exchanged.
    `numpy_module` carries a numpy-like (numpy, cupy) module that will be
    used to direct the exchange on the right device.
    """

    n_points: int
    strides: Tuple[int]
    itemsize: int
    shape: Tuple[int]
    origin: Tuple[int, ...]
    extent: Tuple[int, ...]
    dims: Tuple[str, ...]
    numpy_module: NumpyModule
    dtype: Any

    @classmethod
    def from_quantity(cls, quantity: Quantity, n_points: int) -> "QuantityHaloSpec":
        return QuantityHaloSpec(
            n_points=n_points,
            strides=quantity.data.strides,
            itemsize=quantity.data.itemsize,
            shape=quantity.data.shape,
            origin=quantity.origin,
            extent=quantity.extent,
            dims=quantity.dims,
            numpy_module=quantity.np,
            dtype=quantity.data.dtype,
        )
