from .parallel_translate import ParallelTranslate2PyState
import fv3.stencils.fv_dynamics as fv_dynamics
import copy


class TranslateFVDynamics_Preamble(ParallelTranslate2PyState):
    inputs = {}

    def __init__(self, grids):
        super().__init__(grids)
        self._base.compute_func = fv_dynamics.compute_preamble
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
            "ph1": {},
            "ph2": {},
            "pe": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
                "kaxis": 1,
            },
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
            "dp1": {},
            "ua": {},
            "va": {},
            "ak": {},
            "bk": {},
            "pfull": {},
            # "te_2d":{},
            "cappa": {},
            "cvm": {"kstart": grid.is_, "axis": 0},
        }
        self._base.in_vars["parameters"] = [
            "bdt",
            "ptop",
            "do_adiabatic_init",
            "consv_te",
            "zvir",
        ]
        self._base.out_vars = copy.copy(self._base.in_vars["data_vars"])
        for var in ["ak", "bk", "cappa", "cvm", "ph1", "ph2"]:
            del self._base.out_vars[var]
        self._base.out_vars["phis"] = {"kstart": 0, "kend": 0}
        self._base.out_vars["pfull"] = {
            "istart": grid.is_,
            "iend": grid.is_,
            "jstart": grid.js,
            "jend": grid.js,
        }
