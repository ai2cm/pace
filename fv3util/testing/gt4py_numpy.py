"""This module provides a numpy-style wrapper around certain gt4py functions"""
import functools
try:
    import gt4py
except ImportError:
    gt4py = None
import numpy as np

__all__ = ['inject_storage_methods', 'inject_attrs']


def wrap_storage_call(function, backend):
    def wrapped(shape, dtype=float):
        return function(backend, [0] * len(shape), shape, dtype)
    wrapped.__name__ = function.__name__
    return wrapped


def inject_storage_methods(attr_dict, backend):
    if gt4py is not None:
        attr_dict['zeros'] = wrap_storage_call(gt4py.storage.zeros, backend)
        attr_dict['ones'] = wrap_storage_call(gt4py.storage.ones, backend)
        attr_dict['empty'] = wrap_storage_call(gt4py.storage.empty, backend)


def wrap_call(function):
    name = function.__name__
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        return getattr(np, name)(*args, **kwargs)
    return wrapped


def is_callable(maybe_function):
    return hasattr(maybe_function, '__call__')


def inject_attrs(attr_dict, module):
    for name in set(dir(module)).difference(attr_dict.keys()):
        attr_dict[name] = getattr(module, name)


inject_storage_methods(locals(), 'numpy')
inject_attrs(locals(), np)
