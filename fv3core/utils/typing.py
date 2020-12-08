from typing import Tuple, Union

import gt4py.gtscript as gtscript
import numpy as np


# A Field
Field = gtscript.Field
"""A gt4py field"""

# Axes
IJK = gtscript.IJK
IJ = gtscript.IJ
IK = gtscript.IK
JK = gtscript.JK
I = gtscript.I
J = gtscript.J
K = gtscript.K

# Union of valid data types (from gt4py.gtscript)
DTypes = Union[bool, np.bool, int, np.int32, np.int64, float, np.float32, np.float64]

# Default float and int types
float_type = np.float_
int_type = np.int_
bool_type = np.bool_


class _FieldDescriptor:
    def __init__(self, dtype):
        self.dtype = dtype

    def __getitem__(self, axes):
        return Field[self.dtype, axes]


def _FieldDescriptorMaker(dtype):
    return _FieldDescriptor(dtype)


FloatField = _FieldDescriptor(float_type)
IntField = _FieldDescriptor(int_type)
BoolField = _FieldDescriptor(bool_type)

Index3D = Tuple[int, int, int]
