import pace.dsl
import pace.util
from pace.fv3core.stencils.pk3_halo import PK3Halo
from pace.fv3core.testing import TranslateDycoreFortranData2Py


class TranslatePK3_Halo(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.stencil_factory = stencil_factory
        self.compute_func = PK3Halo(self.stencil_factory)
        self.in_vars["data_vars"] = {"pk3": {}, "delp": {}}
        self.in_vars["parameters"] = ["akap", "ptop"]
        self.out_vars = {"pk3": {"kend": grid.npz + 1}}
