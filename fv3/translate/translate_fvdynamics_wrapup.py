from .parallel_translate import ParallelTranslate2PyState
import fv3.stencils.fv_dynamics as fv_dynamics
from .translate_cubedtolatlon import TranslateCubedToLatLon
import copy
import fv3.utils.gt4py_utils as utils


class TranslateFVDynamics_Wrapup(ParallelTranslate2PyState):
    inputs = TranslateCubedToLatLon.inputs

    def __init__(self, grids):
        super().__init__(grids)
        self._base.compute_func = fv_dynamics.wrapup
        grid = grids[0]
        self._base.in_vars["data_vars"] = {
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "qcld": {},
            "peln": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz,
                "kaxis": 1,
            },
            "delp": {},
            "delz": {},
            "pt": {},
            "ua": {},
            "va": {},
        }
        self._base.in_vars["parameters"] = ["nq"]
        self._base.out_vars = copy.copy(self._base.in_vars["data_vars"])
        # could do this, but don't have to
        # for var in ['delp', 'delz']:
        #    del self._base.out_vars[var]
        for qvar in utils.tracer_variables:
            self.ignore_near_zero_errors[qvar] = True
