from .translate import TranslateFortranData2Py
import fv3core.stencils.pk3_halo as pk3_halo


class TranslatePK3_Halo(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = pk3_halo.compute
        self.in_vars["data_vars"] = {"pk3": {}, "delp": {}}
        self.in_vars["parameters"] = ["akap", "ptop"]
        self.out_vars = {"pk3": {"kend": grid.npz + 1}}
