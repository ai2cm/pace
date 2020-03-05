import fv3.stencils.delnflux as delnflux
from .translate_d_sw import TranslateD_SW


class TranslateDel6VtFlux(TranslateD_SW):
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
        # nord_column = [int(i) for i in inputs['nord_w'][0, 0, :]]

        # self.make_storage_data_input_vars(inputs)
        # del inputs['nord_column']
        # inputs['damp_c'] = inputs['damp_c'][0, 0, 3]
        # delnflux.compute_del6vflux(inputs, nord_column)
        # return self.slice_output(inputs)
        # if 'mass' not in inputs:
        #    inputs['mass'] = None
        return self.nord_column_split_compute(inputs, delnflux.compute_no_sg)
