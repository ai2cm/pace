from .parallel_translate import ParallelTranslate2Py
import fv3.stencils.c2l_ord as c2l_ord
import fv3util


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
        self._base.compute_func = c2l_ord.compute_cubed_to_latlon
        self._base.in_vars["data_vars"] = {"u": {}, "v": {}, "ua": {}, "va": {}}
        self._base.in_vars["parameters"] = ["mode"]
        self._base.out_vars = {
            "ua": {},
            "va": {},
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
        }
