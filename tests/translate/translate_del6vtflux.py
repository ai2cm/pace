import fv3core.stencils.delnflux as delnflux
from fv3core.testing import TranslateFortranData2Py


class TranslateDel6VtFlux(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        fxstat = grid.x3d_domain_dict()
        fxstat.update({"serialname": "fx2"})
        fystat = grid.y3d_domain_dict()
        fystat.update({"serialname": "fy2"})
        self.in_vars["data_vars"] = {
            "q": {"serialname": "wq"},
            "d2": {"serialname": "wd2"},
            "fx2": grid.x3d_domain_dict(),
            "fy2": grid.y3d_domain_dict(),
            "damp_c": {"serialname": "damp4"},
            "nord_column": {"serialname": "nord_w"},
        }
        self.in_vars["parameters"] = []
        self.out_vars = {
            "fx2": grid.x3d_domain_dict(),
            "fy2": grid.y3d_domain_dict(),
            "d2": {"serialname": "wd2"},
            "q": {"serialname": "wq"},
        }

    # use_sg -- 'dx', 'dy', 'rdxc', 'rdyc', 'sin_sg needed
    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        self.compute_func = delnflux.DelnFluxNoSG(inputs.pop("nord_column"))
        self.compute_func(**inputs)
        return self.slice_output(inputs)
