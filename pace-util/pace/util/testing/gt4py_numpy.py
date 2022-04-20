"""This module provides a numpy-style wrapper around certain gt4py functions"""
try:
    import gt4py
except ImportError:
    gt4py = None
import numpy as np


def _wrap_storage_call(function, backend):
    def wrapped(shape, dtype=float):
        return function(backend, [0] * len(shape), shape, dtype, managed_memory=True)

    wrapped.__name__ = function.__name__
    return wrapped


def inject_storage_methods(attr_dict, backend):
    if gt4py is not None:
        attr_dict["zeros"] = _wrap_storage_call(gt4py.storage.zeros, backend)
        attr_dict["ones"] = _wrap_storage_call(gt4py.storage.ones, backend)
        attr_dict["empty"] = _wrap_storage_call(gt4py.storage.empty, backend)


def inject_attrs(attr_dict, module):
    for name in set(dir(module)).difference(attr_dict.keys()):
        attr_dict[name] = getattr(module, name)


inject_storage_methods(locals(), "gtc:numpy")
inject_attrs(locals(), np)
