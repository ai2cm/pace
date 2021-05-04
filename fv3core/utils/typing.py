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
I = gtscript.I  # noqa: E741
J = gtscript.J  # noqa: E741
K = gtscript.K  # noqa: E741

# Union of valid data types (from gt4py.gtscript)
DTypes = Union[bool, np.bool, int, np.int32, np.int64, float, np.float32, np.float64]

# Default float and int types
Float = np.float_
Int = np.int_
Bool = np.bool_


class _FieldDescriptor:
    def __init__(self, dtype):
        self.dtype = dtype

    def __getitem__(self, axes):
        return Field[self.dtype, axes]


def _FieldDescriptorMaker(dtype):
    return _FieldDescriptor(dtype)


FloatField = Field[Float]
FloatFieldI = Field[Float, gtscript.I]
FloatFieldJ = Field[Float, gtscript.J]
FloatFieldIJ = Field[Float, gtscript.IJ]
FloatFieldK = Field[Float, gtscript.K]
IntField = Field[Int]
IntFieldIJ = Field[Int, gtscript.IJ]
BoolField = _FieldDescriptor(Bool)

Index3D = Tuple[int, int, int]
