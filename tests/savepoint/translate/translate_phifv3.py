from fv3gfs.physics.stencils.get_phi_fv3 import get_phi_fv3
from fv3gfs.physics.testing import TranslatePhysicsFortranData2Py
import fv3core._config as spec
from fv3core.decorators import FrozenStencil


class TranslatePhiFV3(TranslatePhysicsFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)

        self.in_vars["data_vars"] = {
            "gt0": {"serialname": "phi_gt0"},
            "gq0": {"serialname": "phi_gq0"},
            "del_gz": {
                "serialname": "phi_del_gz",
                "in_roll_zero": True,
                "out_roll_zero": True,
            },
            "phii": {"serialname": "phi_phii"},
            "phil": {"serialname": "phi_phil", "kend": grid.npz - 1},
        }
        self.out_vars = {
            "del_gz": self.in_vars["data_vars"]["del_gz"],
            "phii": self.in_vars["data_vars"]["phii"],
            "phil": self.in_vars["data_vars"]["phil"],
        }
        self.compute_func = FrozenStencil(
            get_phi_fv3,
            origin=self.grid.grid_indexing.origin_full(),
            domain=self.grid.grid_indexing.domain_full(add=(0, 0, 1)),
        )

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        inputs["gq0"] = inputs["gq0"]["qvapor"]
        self.compute_func(**inputs)
        return self.slice_output(inputs)

