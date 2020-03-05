from .translate import TranslateFortranData2Py
from ..utils.corners import fill_corners_dgrid
import fv3.utils.gt4py_utils as utils
import numpy as np


class TranslateFillCornersVector(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"vc": {}, "uc": {}, "nord_col": {}}
        self.out_vars = {"vc": grid.y3d_domain_dict(), "uc": grid.x3d_domain_dict()}

    def compute(self, inputs):
        nord_column = inputs["nord_col"][0, 0, :]
        vector = True
        self.make_storage_data_input_vars(inputs)
        num_k = self.grid.npz
        for nord in np.unique(nord_column):
            if nord != 0:
                ki = [i for i in range(num_k) if nord_column[i] == nord]
                d = utils.k_slice(inputs, ki)
                self.grid.npz = len(ki)
                fill_corners_dgrid(d["vc"], d["uc"], self.grid, vector)
                inputs["vc"][:, :, ki] = d["vc"]
                inputs["uc"][:, :, ki] = d["uc"]
        self.grid.npz = num_k
        return self.slice_output(inputs)
