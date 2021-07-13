from fv3core.stencils.fxadv import FiniteVolumeFluxPrep
from fv3core.testing import TranslateFortranData2Py


class TranslateFxAdv(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        utinfo = grid.x3d_domain_dict()
        vtinfo = grid.y3d_domain_dict()
        self.compute_func = FiniteVolumeFluxPrep(
            self.grid.grid_indexing,
            self.grid.grid_data,
        )
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "ut": utinfo,
            "vt": vtinfo,
            "x_area_flux": {
                **{"serialname": "xfx_adv"},
                **grid.x3d_compute_domain_y_dict(),
            },
            "crx": {**{"serialname": "crx_adv"}, **grid.x3d_compute_domain_y_dict()},
            "y_area_flux": {
                **{"serialname": "yfx_adv"},
                **grid.y3d_compute_domain_x_dict(),
            },
            "cry": {**{"serialname": "cry_adv"}, **grid.y3d_compute_domain_x_dict()},
        }
        self.in_vars["parameters"] = ["dt"]
        self.out_vars = {
            "ut": utinfo,
            "vt": vtinfo,
        }
        for var in ["x_area_flux", "crx", "y_area_flux", "cry"]:
            self.out_vars[var] = self.in_vars["data_vars"][var]

    def compute_from_storage(self, inputs):
        self.compute_func(**inputs)
        return inputs
