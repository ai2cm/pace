from fv3core.decorators import FrozenStencil
from fv3gfs.physics.stencils.update_atmos_state import (
    prepare_tendencies_and_update_tracers,
)
from fv3gfs.physics.testing import TranslatePhysicsFortranData2Py


class TranslatePhysUpdateTracers(TranslatePhysicsFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "u_dt": {"dycore": True},
            "v_dt": {"dycore": True},
            "pt_dt": {"serialname": "t_dt", "dycore": True},
            "u_t1": {"order": "F"},
            "v_t1": {"order": "F"},
            "pt_t1": {"order": "F"},
            "qvapor_t1": {"order": "F"},
            "qliquid_t1": {"order": "F"},
            "qrain_t1": {"order": "F"},
            "qsnow_t1": {"order": "F"},
            "qice_t1": {"order": "F"},
            "qgraupel_t1": {"order": "F"},
            "u_t0": {"order": "F"},
            "v_t0": {"order": "F"},
            "pt_t0": {"order": "F"},
            "qvapor_t0": {"dycore": True, "kend": grid.npz - 1},
            "qliquid_t0": {"dycore": True, "kend": grid.npz - 1},
            "qrain_t0": {"dycore": True, "kend": grid.npz - 1},
            "qsnow_t0": {"dycore": True, "kend": grid.npz - 1},
            "qice_t0": {"dycore": True, "kend": grid.npz - 1},
            "qgraupel_t0": {"dycore": True, "kend": grid.npz - 1},
            "prsi": {"serialname": "IPD_prsi", "order": "F"},
            "delp": {"dycore": True, "serialname": "IPD_delp", "kend": grid.npz - 1},
        }
        self.in_vars["parameters"] = ["rdt"]
        self.out_vars = {
            "u_dt": {
                "dycore": True,
                "compute": False,
                "kend": grid.npz - 1,
            },
            "v_dt": {
                "dycore": True,
                "compute": False,
                "kend": grid.npz - 1,
            },
            "pt_dt": {
                "serialname": "t_dt",
                "dycore": True,
                "kend": grid.npz - 1,
            },
            "delp": {
                "dycore": True,
                "kend": grid.npz - 1,
                "compute": False,
                "out_roll_zero": True,
            },
            "qvapor_t0": {"dycore": True, "kend": grid.npz - 1, "compute": False},
            "qliquid_t0": {"dycore": True, "kend": grid.npz - 1, "compute": False},
            "qrain_t0": {"dycore": True, "kend": grid.npz - 1, "compute": False},
            "qsnow_t0": {"dycore": True, "kend": grid.npz - 1, "compute": False},
            "qice_t0": {"dycore": True, "kend": grid.npz - 1, "compute": False},
            "qgraupel_t0": {"dycore": True, "kend": grid.npz - 1, "compute": False},
        }
        self.compute_func = FrozenStencil(
            prepare_tendencies_and_update_tracers,
            origin=self.grid.grid_indexing.origin_compute(),
            domain=self.grid.grid_indexing.domain_compute(add=(0, 0, 1)),
        )

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        self.compute_func(**inputs)
        return self.slice_output(inputs)
