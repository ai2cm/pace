import fv3core.stencils.fxadv as fxadv

from .translate import TranslateFortranData2Py


class TranslateFxAdv(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        utinfo = grid.x3d_domain_dict()
        utinfo["serialname"] = "ut"
        vtinfo = grid.y3d_domain_dict()
        vtinfo["serialname"] = "vt"
        # TODO: Do we want this to be bit reproducible? We think this error is
        # spawning from generalizing the u and v corner calculations, ut it's
        # possible we are missing something
        self.max_error = 1e-12
        self.in_vars["data_vars"] = {
            "uc_in": {"serialname": "uc"},
            "vc_in": {"serialname": "vc"},
            "ut_in": utinfo,
            "vt_in": vtinfo,
            "xfx_adv": grid.x3d_compute_domain_y_dict(),
            "crx_adv": grid.x3d_compute_domain_y_dict(),
            "yfx_adv": grid.y3d_compute_domain_x_dict(),
            "cry_adv": grid.y3d_compute_domain_x_dict(),
        }
        self.in_vars["parameters"] = ["dt"]
        self.out_vars = {
            "ra_x": {"istart": grid.is_, "iend": grid.ie},
            "ra_y": {"jstart": grid.js, "jend": grid.je},
        }
        for invar, info in self.in_vars["data_vars"].items():
            if "c_in" not in invar:
                self.out_vars[invar] = info
        # TODO: There is roundoff error at 1e-15 max_error due to different
        # ordering of additive terms in the y direction, as a result of reusing
        # the same stencil for x and y directions

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        ra_x, ra_y = fxadv.compute(**inputs)
        return self.slice_output(inputs, {"ra_x": ra_x, "ra_y": ra_y})
