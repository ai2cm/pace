import os

import numpy as np
from gt4py import gtscript


BACKEND = str(os.getenv("BACKEND")) if ("BACKEND" in os.environ) else "numpy"
REBUILD = (os.getenv("REBUILD") == "True") if ("REBUILD" in os.environ) else True
IS_DOCKER = (os.getenv("IS_DOCKER") == "True") if ("IS_DOCKER" in os.environ) else True
DTYPE_INT = np.int32
FIELD_INT = gtscript.Field[DTYPE_INT]
DTYPE_FLT = np.float64
FIELD_FLT = gtscript.Field[DTYPE_FLT]
FIELD_FLTIJ = gtscript.Field[gtscript.IJ, DTYPE_FLT]
FIELD_FLTI = gtscript.Field[gtscript.I, DTYPE_FLT]
FIELD_FLTJ = gtscript.Field[gtscript.J, DTYPE_FLT]
DEFAULT_ORIGIN = (0, 0, 0)
HALO = 3
# Path of serialbox directory
if IS_DOCKER:
    SERIALBOX_DIR = "/usr/local/serialbox"
else:
    SERIALBOX_DIR = "/project/c14/install/daint/serialbox2_master/gnu_debug"
