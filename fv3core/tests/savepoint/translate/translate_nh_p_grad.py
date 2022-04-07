import fv3core.stencils.nh_p_grad as NH_P_Grad
from pace.stencils.testing import TranslateDycoreFortranData2Py


class TranslateNH_P_Grad(TranslateDycoreFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
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
        self.stencil_factory = stencil_factory
        self.namelist = namelist

    def compute(self, inputs):
        self.compute_func = NH_P_Grad.NonHydrostaticPressureGradient(
            self.stencil_factory, self.grid.grid_data, self.namelist.grid_type
        )
        self.make_storage_data_input_vars(inputs)
        self.compute_func(**inputs)
        return self.slice_output(inputs)
