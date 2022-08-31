import pace.dsl
import pace.util
from pace.fv3core import DynamicalCoreConfig
from pace.fv3core.stencils.saturation_adjustment import SatAdjust3d
from pace.fv3core.testing import TranslateDycoreFortranData2Py


class TranslateSatAdjust3d(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "te": {},
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "qcld": {},
            "hs": {},
            "peln": {"istart": grid.is_, "jstart": grid.js, "kaxis": 1},
            "delp": {},
            "delz": {},
            "q_con": {},
            "pt": {},
            "pkz": {"istart": grid.is_, "jstart": grid.js},
            "cappa": {},
        }
        self.max_error = 5e-14
        # te0 is off by 1e-10 when you do nothing...
        self.in_vars["parameters"] = [
            "r_vir",
            "mdt",
            "fast_mp_consv",
            "last_step",
            "akap",
            "kmp",
        ]
        self.out_vars = {
            "te": {},
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "qcld": {},
            "q_con": {},
            "pt": {},
            "pkz": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
            "cappa": {},
        }
        self.namelist = namelist
        self.stencil_factory = stencil_factory

    def compute_from_storage(self, inputs):
        inputs["kmp"] -= 1
        inputs["last_step"] = bool(inputs["last_step"])
        inputs["fast_mp_consv"] = bool(inputs["fast_mp_consv"])
        satadjust3d_obj = SatAdjust3d(
            self.stencil_factory,
            DynamicalCoreConfig.from_namelist(self.namelist).sat_adjust,
            self.grid.area_64,
            int(inputs["kmp"]),
        )
        satadjust3d_obj(**inputs)
        return inputs
