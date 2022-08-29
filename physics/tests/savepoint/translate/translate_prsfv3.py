from pace.physics.stencils.get_prs_fv3 import get_prs_fv3
from pace.stencils.testing.translate_physics import TranslatePhysicsFortranData2Py


class TranslatePrsFV3(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "phii": {"serialname": "prs_phii"},
            "prsi": {"serialname": "prs_prsi"},
            "tgrs": {"serialname": "prs_tgrs"},
            "qgrs": {"serialname": "prs_qgrs"},
            "del_": {"serialname": "prs_del", "kend": namelist.npz - 1},
            "del_gz": {"serialname": "prs_del_gz", "out_roll_zero": True},
        }
        self.out_vars = {
            "del_": self.in_vars["data_vars"]["del_"],
            "del_gz": self.in_vars["data_vars"]["del_gz"],
        }
        self.compute_func = stencil_factory.from_origin_domain(
            get_prs_fv3,
            origin=stencil_factory.grid_indexing.origin_full(),
            domain=stencil_factory.grid_indexing.domain_full(add=(0, 0, 1)),
        )

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        inputs["qgrs"] = inputs["qgrs"]["qvapor"]
        self.compute_func(**inputs)
        return self.slice_output(inputs)
