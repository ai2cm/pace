import tempfile

import numpy as np
import xr

from pace.util.checkpointer import ValidationCheckpointer


def test_validation_basic():
    temp_dir = tempfile.TemporaryDirectory()
    threshold = 1.0

    shape = (6, 6)
    values = np.ones(shape)

    checkpointer = ValidationCheckpointer(
        temp_dir, {"test": {"array": threshold}}, nhalo=0
    )
