import numpy as np

import fv3core.utils.gt4py_utils as utils
from fv3core.testing import TranslateFortranData2Py
from fv3core.utils import corners


class TranslateFill4Corners(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"q4c": {}}
        self.in_vars["parameters"] = ["dir"]
        self.out_vars = {"q4c": {}}

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        corners.fill_corners_cells(inputs["q4c"], "x" if inputs["dir"] == 1 else "y")
        return self.slice_output(inputs, {"q4c": inputs["q4c"]})


class TranslateFillCorners(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"divg_d": {}, "nord_col": {}}
        self.in_vars["parameters"] = ["dir"]
        self.out_vars = {"divg_d": {"iend": grid.ied + 1, "jend": grid.jed + 1}}

    def compute(self, inputs):
        if inputs["dir"] == 1:
            direction = "x"
        elif inputs["dir"] == 2:
            direction = "y"
        else:
            raise ValueError("Invalid input")
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
                corners.fill_corners_2d(d["divg_d"], self.grid, "B", direction)
                inputs["divg_d"][:, :, ki] = d["divg_d"]
        self.grid.npz = num_k
        return self.slice_output(inputs)


class TranslateCopyCorners(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"q": {}}
        self.in_vars["parameters"] = ["dir"]
        self.out_vars = {"q": {}}

    def compute(self, inputs):
        if inputs["dir"] == 1:
            direction = "x"
        elif inputs["dir"] == 2:
            direction = "y"
        else:
            raise ValueError("Invalid input")
        corners.copy_corners(inputs["q"], direction, self.grid)
        return {"q": inputs["q"]}


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
                corners.fill_corners_dgrid(d["vc"], d["uc"], self.grid, vector)
                inputs["vc"][:, :, ki] = d["vc"]
                inputs["uc"][:, :, ki] = d["uc"]
        self.grid.npz = num_k
        return self.slice_output(inputs)
