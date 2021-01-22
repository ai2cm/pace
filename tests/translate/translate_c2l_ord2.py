import fv3core.stencils.c2l_ord as c2l_ord
from fv3core.testing import TranslateFortranData2Py


class TranslateC2L_Ord2(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"u": {}, "v": {}, "ua": {}, "va": {}}
        self.in_vars["parameters"] = []  # do_halo
        self.out_vars = {"ua": {}, "va": {}}

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        c2l_ord.c2l_ord2(
            **inputs,
            dx=self.grid.dx,
            dy=self.grid.dy,
            a11=self.grid.a11,
            a12=self.grid.a12,
            a21=self.grid.a21,
            a22=self.grid.a22,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(),
        )
        return self.slice_output(inputs)
