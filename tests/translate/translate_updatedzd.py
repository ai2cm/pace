import fv3core.stencils.updatedzd as updatedzd
from fv3core.testing import TranslateFortranData2Py


class TranslateUpdateDzD(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = updatedzd.compute
        self.in_vars["data_vars"] = {
            "ndif": {},  # column var
            "damp_vtd": {},  # column var
            "dp0": {},  # column var
            "zs": {},
            "zh": {"kend": grid.npz + 1},
            "crx": grid.x3d_compute_domain_y_dict(),
            "cry": grid.y3d_compute_domain_x_dict(),
            "xfx": grid.x3d_compute_domain_y_dict(),
            "yfx": grid.y3d_compute_domain_x_dict(),
            "wsd": grid.compute_dict(),
        }

        self.in_vars["parameters"] = ["dt"]
        out_vars = ["zh", "crx", "cry", "xfx", "yfx", "wsd"]
        self.out_vars = {}
        for v in out_vars:
            self.out_vars[v] = self.in_vars["data_vars"][v]
        self.out_vars["wsd"]["kstart"] = grid.npz
        self.out_vars["wsd"]["kend"] = None

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        inputs["ndif"] = inputs["ndif"][0, 0, :]
        inputs["damp_vtd"] = inputs["damp_vtd"][0, 0, :]
        self.compute_func(**inputs)
        return self.slice_output(inputs)
