from .translate import TranslateFortranData2Py
import fv3.stencils.remapping_part2 as remap_part2
import fv3.utils.gt4py_utils as utils


class TranslateRemapping_Part2(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = remap_part2.compute
        self.in_vars["data_vars"] = {
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "qcld": {},
            "w": {},
            "u": grid.y3d_domain_dict(),
            "ua": {},
            "v": grid.x3d_domain_dict(),
            "delz": {},
            "pt": {},
            "delp": {},
            "cappa": {},
            "q_con": {},
            "gz": {"serialname": "gz1d", "kstart": grid.is_, "axis": 0},
            "cvm": {"kstart": grid.is_, "axis": 0},
            "pkz": grid.compute_dict(),
            "pk": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz,
            },
            "peln": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kaxis": 1,
                "kend": grid.npz,
            },
            "pe": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
                "kaxis": 1,
            },
            "hs": {},
            "pfull": {},
            "te_2d": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kstart": grid.npz - 1,
                "kend": grid.npz - 1,
            },
            "te0_2d": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kstart": grid.npz - 1,
                "kend": grid.npz - 1,
            },
            "te": {},
            "zsum1": {
                "istart": grid.is_,
                "jstart": grid.js,
                "iend": grid.ie,
                "jend": grid.je,
                "kstart": grid.npz - 1,
                "kend": grid.npz - 1,
            },
        }
        self.in_vars["parameters"] = [
            "ptop",
            "akap",
            "r_vir",
            "last_step",
            "pdt",
            "mdt",
            "consv",
            "do_adiabatic_init",
        ]
        self.out_vars = {}
        for k in [
            "pe",
            "pkz",
            "pk",
            "peln",
            "pt",
            "qvapor",
            "qliquid",
            "qice",
            "qrain",
            "qsnow",
            "qgraupel",
            "qcld",
            "cappa",
            "delp",
            "delz",
            "q_con",
            "te",
            "te_2d",
            "te0_2d",
            "zsum1",
        ]:
            self.out_vars[k] = self.in_vars["data_vars"][k]
        self.max_error = 2e-14
