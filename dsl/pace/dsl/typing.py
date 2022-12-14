from typing import Tuple, Union, cast

import gt4py.cartesian.gtscript as gtscript
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

# Union of valid data types (from gt4py.cartesian.gtscript)
DTypes = Union[bool, np.bool_, int, np.int32, np.int64, float, np.float32, np.float64]

# Default float and int types
Float = np.float_
Int = np.int_
Bool = np.bool_


FloatField = Field[gtscript.IJK, Float]
FloatFieldI = Field[gtscript.I, Float]
FloatFieldJ = Field[gtscript.J, Float]
FloatFieldIJ = Field[gtscript.IJ, Float]
FloatFieldK = Field[gtscript.K, Float]
IntField = Field[gtscript.IJK, Int]
IntFieldIJ = Field[gtscript.IJ, Int]
IntFieldK = Field[gtscript.K, Int]
BoolField = Field[gtscript.IJK, Bool]

Index3D = Tuple[int, int, int]


def cast_to_index3d(val: Tuple[int, ...]) -> Index3D:
    if len(val) != 3:
        raise ValueError(f"expected 3d index, received {val}")
    return cast(Index3D, val)
