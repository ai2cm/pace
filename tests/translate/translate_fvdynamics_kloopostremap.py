from .parallel_translate import ParallelTranslate2PyState
import fv3core.stencils.fv_dynamics as fv_dynamics
from .translate_dyncore import TranslateDynCore
from .translate_tracer2d1l import TranslateTracer2D1L
import fv3core.utils.gt4py_utils as utils

import fv3gfs.util as fv3util
import copy


class TranslateFVDynamics(ParallelTranslate2PyState):
    inputs = {**TranslateDynCore.inputs, **TranslateTracer2D1L.inputs}
    del inputs["cappa"]

    def __init__(self, grids):
        super().__init__(grids)
        self._base.compute_func = fv_dynamics.compute
        grid = grids[0]
        self._base.in_vars["data_vars"] = {
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "w": {},
            "delz": {},
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "qcld": {},
            "ps": {},
            "pe": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
                "kaxis": 1,
            },
            "pk": grid.compute_buffer_k_dict(),
            "peln": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz,
                "kaxis": 1,
            },
            "pkz": grid.compute_dict(),
            "phis": {},
            "q_con": {},
            "delp": {},
            "pt": {},
            "omga": {},
            "ua": {},
            "va": {},
            "uc": grid.x3d_domain_dict(),
            "vc": grid.y3d_domain_dict(),
            "ak": {},
            "bk": {},
            "mfxd": grid.x3d_compute_dict(),
            "mfyd": grid.y3d_compute_dict(),
            "cxd": grid.x3d_compute_domain_y_dict(),
            "cyd": grid.y3d_compute_domain_x_dict(),
            "diss_estd": {},
        }
        self._base.in_vars["parameters"] = [
            "bdt",
            "zvir",
            "ptop",
            "ks",
            "n_split",
            "nq_tot",
            "do_adiabatic_init",
            "consv_te",
        ]
        self._base.out_vars = copy.copy(self._base.in_vars["data_vars"])
        self.max_error = 1e-4
        for var in ["ak", "bk"]:
            del self._base.out_vars[var]
        self._base.out_vars["ps"] = {"kstart": grid.npz - 1, "kend": grid.npz - 1}
        self._base.out_vars["phis"] = {"kstart": 0, "kend": 0}

        # w, 17,10 0
