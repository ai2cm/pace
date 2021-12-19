from fv3core.stencils.fxadv import FiniteVolumeFluxPrep
from pace.stencils.testing import TranslateFortranData2Py


class TranslateFxAdv(TranslateFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        utinfo = grid.x3d_domain_dict()
        utinfo["serialname"] = "ut"
        vtinfo = grid.y3d_domain_dict()
        vtinfo["serialname"] = "vt"
        self.stencil_factory = stencil_factory
        self.compute_func = FiniteVolumeFluxPrep(
            self.stencil_factory,
            self.grid.grid_data,
        )
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "uc_contra": utinfo,
            "vc_contra": vtinfo,
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
            "uc_contra": utinfo,
            "vc_contra": vtinfo,
        }
        for var in ["x_area_flux", "crx", "y_area_flux", "cry"]:
            self.out_vars[var] = self.in_vars["data_vars"][var]

    def compute_from_storage(self, inputs):
        self.compute_func(**inputs)
        return inputs
