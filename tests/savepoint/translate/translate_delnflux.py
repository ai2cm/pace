import fv3core.stencils.delnflux as delnflux
from fv3core.testing import TranslateFortranData2Py


class TranslateDelnFlux(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
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
        self.make_storage_data_input_vars(inputs)
        self.compute_func = delnflux.DelnFlux(
            inputs.pop("nord_column"), inputs.pop("damp_c")
        )
        self.compute_func(**inputs)
        return self.slice_output(inputs)


class TranslateDelnFlux_2(TranslateDelnFlux):
    def __init__(self, grid):
        super().__init__(grid)
        del self.in_vars["data_vars"]["mass"]
