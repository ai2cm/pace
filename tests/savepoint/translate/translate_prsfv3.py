from fv3core.decorators import FrozenStencil
from fv3gfs.physics.stencils.get_prs_fv3 import get_prs_fv3
from fv3gfs.physics.testing import TranslatePhysicsFortranData2Py


class TranslatePrsFV3(TranslatePhysicsFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)

        self.in_vars["data_vars"] = {
            "phii": {"serialname": "prs_phii"},
            "prsi": {"serialname": "prs_prsi"},
            "tgrs": {"serialname": "prs_tgrs"},
            "qgrs": {"serialname": "prs_qgrs"},
            "del_": {"serialname": "prs_del", "kend": grid.npz - 1},
            "del_gz": {"serialname": "prs_del_gz", "out_roll_zero": True},
        }
        self.out_vars = {
            "del_": self.in_vars["data_vars"]["del_"],
            "del_gz": self.in_vars["data_vars"]["del_gz"],
        }
        self.compute_func = FrozenStencil(
            get_prs_fv3,
            origin=self.grid.grid_indexing.origin_full(),
            domain=self.grid.grid_indexing.domain_full(add=(0, 0, 1)),
        )

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        inputs["qgrs"] = inputs["qgrs"]["qvapor"]
        self.compute_func(**inputs)
        return self.slice_output(inputs)
