import pace.dsl
import pace.util
from pace.fv3core.stencils.d2a2c_vect import DGrid2AGrid2CGridVectors
from pace.fv3core.testing import TranslateDycoreFortranData2Py


class TranslateD2A2C_Vect(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        dord4 = True
        self.stencil_factory = stencil_factory
        self.namelist = namelist  # type: ignore
        self.compute_func = DGrid2AGrid2CGridVectors(  # type: ignore
            self.stencil_factory,
            self.grid.grid_data,
            self.grid.nested,
            self.namelist.grid_type,
            dord4,
        )
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "u": {},
            "v": {},
            "ua": {},
            "va": {},
            "utc": {},
            "vtc": {},
        }
        self.out_vars = {
            "uc": grid.x3d_domain_dict(),
            "vc": grid.y3d_domain_dict(),
            "ua": {},
            "va": {},
            "utc": {},
            "vtc": {},
        }
        # TODO: This seems to be needed primarily for the edge_interpolate_4
        # methods, can we rejigger the order of operations to make it match to
        # more precision?
        self.max_error = 2e-10
