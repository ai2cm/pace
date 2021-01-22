import fv3core.stencils.c2l_ord as c2l_ord
import fv3gfs.util as fv3util
from fv3core.testing import ParallelTranslate2Py


class TranslateCubedToLatLon(ParallelTranslate2Py):
    inputs = {
        "u": {
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "v": {
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
    }

    def __init__(self, grids):
        super().__init__(grids)
        grid = grids[0]
        self._base.compute_func = self.do_compute
        self._base.in_vars["data_vars"] = {"u": {}, "v": {}, "ua": {}, "va": {}}
        self._base.out_vars = {
            "ua": {},
            "va": {},
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
        }

    def do_compute(self, **kwargs):
        c2l_ord.compute_cubed_to_latlon(**kwargs, do_halo_update=True)
