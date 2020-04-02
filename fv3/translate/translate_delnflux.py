import fv3.stencils.delnflux as delnflux
from fv3.translate.translate import TranslateFortranData2Py
import fv3.utils.gt4py_utils as utils


class TranslateDelnFlux(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = delnflux.compute_delnflux_no_sg
        self.in_vars["data_vars"] = {
            "q": {},
            "fx": grid.x3d_compute_dict(),
            "fy": grid.y3d_compute_dict(),
            "damp_c": {},
            "nord_column": {},
            "mass": {},
        }
        self.in_vars["parameters"] = []
        self.out_vars = {"fx": grid.x3d_compute_dict(), "fy": grid.y3d_compute_dict()}

    # If use_sg is defined -- 'dx', 'dy', 'rdxc', 'rdyc', 'sin_sg needed
    def compute(self, inputs):
        if "mass" not in inputs:
            inputs["mass"] = None
        return self.column_split_compute(
            inputs, {"nord": "nord_column", "damp_c": "damp_c"}
        )


class TranslateDelnFlux_2(TranslateDelnFlux):
    def __init__(self, grid):
        super().__init__(grid)
        del self.in_vars["data_vars"]["mass"]
