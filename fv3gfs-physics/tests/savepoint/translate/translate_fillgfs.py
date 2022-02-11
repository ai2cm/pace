from fv3gfs.physics.testing.translate_physics import TranslatePhysicsFortranData2Py
from pace.stencils.update_atmos_state import fill_gfs


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
        self.compute_func = stencil_factory.from_origin_domain(
            fill_gfs,
            origin=stencil_factory.grid_indexing.origin_full(),
            domain=stencil_factory.grid_indexing.domain_full(add=(0, 0, 1)),
        )

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        inputs["q"] = inputs["q"]["qvapor"]
        inputs["q_min"] = 1.0e-9
        self.compute_func(**inputs)
        return self.slice_output(inputs)
