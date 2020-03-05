from .translate import TranslateFortranData2Py
import fv3.stencils.vorticity_volumemean as vm


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
        vm.compute(**inputs)
        return self.slice_output(inputs)
