import pace.dsl
import pace.dsl.gt4py_utils as utils
import pace.util
from pace.fv3core import DynamicalCoreConfig
from pace.fv3core.stencils.remapping import LagrangianToEulerian
from pace.fv3core.testing import TranslateDycoreFortranData2Py
from pace.util import Z_DIM


class TranslateRemapping(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "tracers": {},
            "w": {},
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "delz": {},
            "pt": {},
            "dp1": {},
            "delp": {},
            "cappa": {},
            "q_con": {},
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
            "hs": {"serialname": "phis"},
            "ps": {},
            "wsd": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
            # column variables...
            "ak": {},
            "bk": {},
            "pfull": {},
        }
        self.in_vars["parameters"] = [
            "ptop",
            "akap",
            "zvir",
            "last_step",
            "consv_te",
            "mdt",
            "nq",
        ]
        self.out_vars = {}
        self.write_vars = ["wsd"]
        for k in [
            "pe",
            "pkz",
            "pk",
            "peln",
            "pt",
            "tracers",
            "cappa",
            "delp",
            "delz",
            "q_con",
            "u",
            "v",
            "w",
            "ps",
            "dp1",
        ]:
            self.out_vars[k] = self.in_vars["data_vars"][k]
        self.out_vars["ps"] = {"kstart": grid.npz, "kend": grid.npz}
        self.max_error = 2e-8
        self.near_zero = 3e-18
        self.ignore_near_zero_errors = {"q_con": True, "tracers": True}
        self.stencil_factory = stencil_factory
        self.namelist = DynamicalCoreConfig.from_namelist(namelist)

    def compute_from_storage(self, inputs):
        wsd_2d = utils.make_storage_from_shape(
            inputs["wsd"].shape[0:2], backend=self.stencil_factory.backend
        )
        wsd_2d[:, :] = inputs["wsd"][:, :, 0]
        inputs["wsd"] = wsd_2d
        inputs["q_cld"] = inputs["tracers"]["qcld"]
        inputs["last_step"] = bool(inputs["last_step"])
        pfull = self.grid.quantity_factory.empty([Z_DIM], units="Pa")
        pfull.data[:] = pfull.np.asarray(inputs.pop("pfull"))
        l_to_e_obj = LagrangianToEulerian(
            self.stencil_factory,
            quantity_factory=self.grid.quantity_factory,
            config=DynamicalCoreConfig.from_namelist(self.namelist).remapping,
            area_64=self.grid.area_64,
            nq=inputs.pop("nq"),
            pfull=pfull,
            tracers=inputs["tracers"],
        )
        l_to_e_obj(**inputs)
        inputs.pop("q_cld")
        return inputs
