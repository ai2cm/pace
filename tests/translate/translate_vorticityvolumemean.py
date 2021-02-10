from fv3core.stencils.d_sw import horizontal_relative_vorticity_from_winds
from fv3core.testing import TranslateFortranData2Py


class TranslateVorticityVolumeMean(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"u": {}, "v": {}, "ut": {}, "vt": {}, "wk": {}}
        self.out_vars = {
            "wk": {},
            "ut": grid.x3d_domain_dict(),
            "vt": grid.y3d_domain_dict(),
        }

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        horizontal_relative_vorticity_from_winds(
            inputs["u"],
            inputs["v"],
            inputs["ut"],
            inputs["vt"],
            self.grid.dx,
            self.grid.dy,
            self.grid.rarea,
            inputs["wk"],
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(),
        )
        return self.slice_output(inputs)
