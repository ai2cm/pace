"""This module provides a cupy-style wrapper around certain gt4py functions"""
try:
    import gt4py
except ImportError:
    gt4py = None
try:
    import cupy
except ImportError:
    cupy = None
from .gt4py_numpy import inject_storage_methods, inject_attrs


if cupy is not None:
    inject_storage_methods(locals(), "cupy")
    inject_attrs(locals(), cupy)
