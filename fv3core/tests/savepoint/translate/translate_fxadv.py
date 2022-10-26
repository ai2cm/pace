import numpy as np

import pace.dsl
import pace.util
from pace.fv3core.stencils.fxadv import FiniteVolumeFluxPrep
from pace.fv3core.testing import TranslateDycoreFortranData2Py
from pace.fv3core.utils.functional_validation import get_subset_func


class TranslateFxAdv(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        utinfo = grid.x3d_domain_dict()
        utinfo["serialname"] = "ut"
        vtinfo = grid.y3d_domain_dict()
        vtinfo["serialname"] = "vt"
        self.stencil_factory = stencil_factory
        self.compute_func = FiniteVolumeFluxPrep(  # type: ignore
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

        self._subset = get_subset_func(
            self.grid.grid_indexing,
            dims=[pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            n_halo=((2, 2), (2, 2)),
        )

    def subset_output(self, varname: str, output: np.ndarray) -> np.ndarray:
        """
        Given an output array, return the slice of the array which we'd
        like to validate against reference data
        """
        if varname in ["uc_contra", "vc_contra", "ut", "vt"]:
            return self._subset(output)
        else:
            return output

    def compute_from_storage(self, inputs):
        self.compute_func(**inputs)
        for name in ["uc_contra", "vc_contra"]:
            inputs[name] = self.subset_output(name, inputs[name])

        return inputs
