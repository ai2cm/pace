from .translate import TranslateFortranData2Py
from ..utils.corners import fill_corners_2d
import fv3.utils.gt4py_utils as utils
import numpy as np


class TranslateFillCorners(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"divg_d": {}, "nord_col": {}}
        self.in_vars["parameters"] = ["dir"]
        self.out_vars = {"divg_d": {"iend": grid.ied + 1, "jend": grid.jed + 1}}

    def compute(self, inputs):
        if inputs["dir"] == 1:
            direction = "x"
        if inputs["dir"] == 2:
            direction = "y"
        # for nord in inputs['nord_col'][0,0,:]:
        #    if nord != 0:
        #        fill_corners_2d(inputs['divg_d'], self.grid, 'B', direction)
        # return {'divg_d':inputs['divg_d']}
        nord_column = inputs["nord_col"][0, 0, :]
        self.make_storage_data_input_vars(inputs)
        num_k = self.grid.npz
        for nord in np.unique(nord_column):
            if nord != 0:
                ki = [i for i in range(num_k) if nord_column[i] == nord]
                d = utils.k_slice(inputs, ki)
                self.grid.npz = len(ki)
                fill_corners_2d(d["divg_d"], self.grid, "B", direction)
                inputs["divg_d"][:, :, ki] = d["divg_d"]
        self.grid.npz = num_k
        return self.slice_output(inputs)
