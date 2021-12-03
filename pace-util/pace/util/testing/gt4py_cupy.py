"""This module provides a cupy-style wrapper around certain gt4py functions"""
try:
    import gt4py
except ImportError:
    gt4py = None
try:
    import cupy
except ImportError:
    cupy = None
import numpy

from .gt4py_numpy import inject_attrs, inject_storage_methods


if cupy is not None:
    inject_storage_methods(locals(), "gtcuda")
    inject_attrs(locals(), cupy)

    def all(a, axis=None, out=None, keepdims=False):
        """Tests whether all array elements along a given axis evaluate to True.
        Args:
            a (gt4py.storage.GPUStorage): Input array.
            axis (int or tuple of ints): Along which axis to compute all.
                The flattened array is used by default.
            out (cupy.ndarray): Output array.
            keepdims (bool): If ``True``, the axis is remained as an axis of
                size one.
        Returns:
            cupy.ndarray: An array reduced of the input array along the axis.
        .. seealso:: :func:`numpy.all`
        """
        return numpy.all(a, axis=axis, out=out, keepdims=keepdims)
