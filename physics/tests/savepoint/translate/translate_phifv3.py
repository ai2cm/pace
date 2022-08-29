from pace.physics.stencils.get_phi_fv3 import get_phi_fv3
from pace.stencils.testing.translate_physics import TranslatePhysicsFortranData2Py


class TranslatePhiFV3(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "gt0": {"serialname": "phi_gt0"},
            "gq0": {"serialname": "phi_gq0"},
            "del_gz": {
                "serialname": "phi_del_gz",
                "in_roll_zero": True,
                "out_roll_zero": True,
            },
            "phii": {"serialname": "phi_phii"},
            "phil": {"serialname": "phi_phil", "kend": namelist.npz - 1},
        }
        self.out_vars = {
            "del_gz": self.in_vars["data_vars"]["del_gz"],
            "phii": self.in_vars["data_vars"]["phii"],
            "phil": self.in_vars["data_vars"]["phil"],
        }
        self.compute_func = stencil_factory.from_origin_domain(
            get_phi_fv3,
            origin=stencil_factory.grid_indexing.origin_full(),
            domain=stencil_factory.grid_indexing.domain_full(add=(0, 0, 1)),
        )

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        inputs["gq0"] = inputs["gq0"]["qvapor"]
        self.compute_func(**inputs)
        return self.slice_output(inputs)
