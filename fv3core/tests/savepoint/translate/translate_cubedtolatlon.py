import fv3core._config as spec
import pace.util as fv3util
from fv3core.testing import ParallelTranslate2Py
from pace.stencils.c2l_ord import CubedToLatLon


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
        spec.set_grid(grid)
        self._base.compute_func = CubedToLatLon(
            grid.stencil_factory, grid.grid_data, order=spec.namelist.c2l_ord
        )
        self._base.in_vars["data_vars"] = {"u": {}, "v": {}, "ua": {}, "va": {}}
        self._base.out_vars = {
            "ua": {},
            "va": {},
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
        }
