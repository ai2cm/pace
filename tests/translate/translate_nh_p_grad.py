import fv3core._config as spec
import fv3core.stencils.nh_p_grad as NH_P_Grad
from fv3core.testing import TranslateFortranData2Py


class TranslateNH_P_Grad(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = NH_P_Grad.NonHydrostaticPressureGradient(
            spec.namelist.grid_type
        )
        self.in_vars["data_vars"] = {
            "u": {},
            "v": {},
            "pp": {},
            "gz": {},
            "pk3": {},
            "delp": {},
        }
        self.in_vars["parameters"] = ["dt", "ptop", "akap"]
        self.out_vars = {
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "pp": {"kend": grid.npz + 1},
            "gz": {"kend": grid.npz + 1},
            "pk3": {"kend": grid.npz + 1},
            "delp": {},
        }

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        self.compute_func(**inputs)
        return self.slice_output(inputs)
