from pace.stencils.testing.translate_physics import TranslatePhysicsFortranData2Py
from pace.stencils.update_atmos_state import prepare_tendencies_and_update_tracers


class TranslatePhysUpdateTracers(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "u_dt": {"dycore": True},
            "v_dt": {"dycore": True},
            "pt_dt": {"serialname": "t_dt", "dycore": True},
            "u_t1": {"order": "F"},
            "v_t1": {"order": "F"},
            "physics_updated_pt": {"serialname": "pt_t1", "order": "F"},
            "physics_updated_specific_humidity": {
                "serialname": "qvapor_t1",
                "order": "F",
            },
            "physics_updated_qliquid": {"serialname": "qliquid_t1", "order": "F"},
            "physics_updated_qrain": {"serialname": "qrain_t1", "order": "F"},
            "physics_updated_qsnow": {"serialname": "qsnow_t1", "order": "F"},
            "physics_updated_qice": {"serialname": "qice_t1", "order": "F"},
            "physics_updated_qgraupel": {"serialname": "qgraupel_t1", "order": "F"},
            "u_t0": {"order": "F"},
            "v_t0": {"order": "F"},
            "pt_t0": {"order": "F"},
            "qvapor_t0": {"dycore": True, "kend": namelist.npz - 1},
            "qliquid_t0": {"dycore": True, "kend": namelist.npz - 1},
            "qrain_t0": {"dycore": True, "kend": namelist.npz - 1},
            "qsnow_t0": {"dycore": True, "kend": namelist.npz - 1},
            "qice_t0": {"dycore": True, "kend": namelist.npz - 1},
            "qgraupel_t0": {"dycore": True, "kend": namelist.npz - 1},
            "prsi": {"serialname": "IPD_prsi", "order": "F"},
            "delp": {
                "dycore": True,
                "serialname": "IPD_delp",
                "kend": namelist.npz - 1,
            },
        }
        self.in_vars["parameters"] = ["rdt"]
        self.out_vars = {
            "u_dt": {
                "dycore": True,
                "compute": False,
                "kend": namelist.npz - 1,
            },
            "v_dt": {
                "dycore": True,
                "compute": False,
                "kend": namelist.npz - 1,
            },
            "pt_dt": {
                "serialname": "t_dt",
                "dycore": True,
                "kend": namelist.npz - 1,
            },
            "delp": {
                "dycore": True,
                "kend": namelist.npz - 1,
                "compute": False,
                "out_roll_zero": True,
            },
            "qvapor_t0": {"dycore": True, "kend": namelist.npz - 1, "compute": False},
            "qliquid_t0": {"dycore": True, "kend": namelist.npz - 1, "compute": False},
            "qrain_t0": {"dycore": True, "kend": namelist.npz - 1, "compute": False},
            "qsnow_t0": {"dycore": True, "kend": namelist.npz - 1, "compute": False},
            "qice_t0": {"dycore": True, "kend": namelist.npz - 1, "compute": False},
            "qgraupel_t0": {"dycore": True, "kend": namelist.npz - 1, "compute": False},
        }
        self.compute_func = stencil_factory.from_origin_domain(
            prepare_tendencies_and_update_tracers,
            origin=stencil_factory.grid_indexing.origin_compute(),
            domain=stencil_factory.grid_indexing.domain_compute(add=(0, 0, 1)),
        )

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        self.compute_func(**inputs)
        return self.slice_output(inputs)
