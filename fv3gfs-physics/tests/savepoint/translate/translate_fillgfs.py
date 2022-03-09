from pace.stencils.testing.translate_physics import TranslatePhysicsFortranData2Py
from pace.stencils.update_atmos_state import fill_gfs_delp
from pace.dsl.typing import FloatField
import pace.dsl.gt4py_utils as utils

def compute_delp(pe:FloatField, delp: FloatField):
    with computation(PARALLEL), interval(0, -1):
        delp = pe[0, 0, 1] - pe[0, 0, 0]
class TranslateFillGFS(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "pe": {"serialname": "IPD_prsi"},
            "q": {"serialname": "IPD_gq0"},
        }
        self.out_vars = {
            "q": {"serialname": "IPD_qvapor", "kend": namelist.npz - 1},
        }
        self.grid_indexing = stencil_factory.grid_indexing
        self.compute_delp = stencil_factory.from_origin_domain(
            compute_delp,
            origin=self.grid_indexing.origin_full(),
            domain=self.grid_indexing.domain_full(add=(1, 1, 1)),
        )
        self.compute_func = stencil_factory.from_origin_domain(
            fill_gfs_delp,
            origin=self.grid_indexing.origin_full(),
            domain=self.grid_indexing.domain_full(add=(0, 0, 1)),
        )

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        inputs["q"] = inputs["q"]["qvapor"]
        inputs["q_min"] = 1.0e-9
        delp = utils.make_storage_from_shape(
            self.grid_indexing.domain_full(add=(1, 1, 1)),
            origin=self.grid_indexing.origin_compute(),
            init=True,
            backend=self.stencil_factory.backend,
        )
        self.compute_delp(inputs["pe"], delp)
        del inputs["pe"]
        inputs["delp"] = delp
        self.compute_func(**inputs)
        return self.slice_output(inputs)
