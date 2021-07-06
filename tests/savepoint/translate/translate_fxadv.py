from fv3core.stencils.fxadv import FiniteVolumeFluxPrep
from fv3core.testing import TranslateFortranData2Py


class TranslateFxAdv(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        utinfo = grid.x3d_domain_dict()
        vtinfo = grid.y3d_domain_dict()
        self.compute_func = FiniteVolumeFluxPrep(
            self.grid.grid_indexing,
            self.grid.dx,
            self.grid.dy,
            self.grid.rdxa,
            self.grid.rdya,
            self.grid.cosa_u,
            self.grid.cosa_v,
            self.grid.rsin_u,
            self.grid.rsin_v,
            self.grid.sin_sg1,
            self.grid.sin_sg2,
            self.grid.sin_sg3,
            self.grid.sin_sg4,
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
