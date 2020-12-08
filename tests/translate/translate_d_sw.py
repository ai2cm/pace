from fv3core.stencils import d_sw

from .translate import TranslateFortranData2Py


class TranslateD_SW(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.max_error = 3e-11  # propagated error from vt roundoff error in FxAdv
        self.in_vars["data_vars"] = {
            "uc": grid.x3d_domain_dict(),
            "vc": grid.y3d_domain_dict(),
            "w": {},
            "delpc": {},
            "delp": {},
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "xfx": grid.x3d_compute_domain_y_dict(),
            "crx": grid.x3d_compute_domain_y_dict(),
            "yfx": grid.y3d_compute_domain_x_dict(),
            "cry": grid.y3d_compute_domain_x_dict(),
            "mfx": grid.x3d_compute_dict(),
            "mfy": grid.y3d_compute_dict(),
            "cx": grid.x3d_compute_domain_y_dict(),
            "cy": grid.y3d_compute_domain_x_dict(),
            "heat_source": {},
            "diss_est": {},
            "q_con": {},
            "pt": {},
            "ptc": {},
            "ua": {},
            "va": {},
            "zh": {},
            "divgd": grid.default_dict_buffer_2d(),
        }
        for name, info in self.in_vars["data_vars"].items():
            info["serialname"] = name + "d"
        self.in_vars["parameters"] = ["dt"]
        self.out_vars = self.in_vars["data_vars"].copy()
        del self.out_vars["zh"]

    # use_sg -- 'dx', 'dy', 'rdxc', 'rdyc', 'sin_sg needed
    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        for v in [
            "utco",
            "vtco",
            "keco",
            "uco",
            "uvort",
            "kex",
            "kevort",
            "vco",
            "ubkey",
            "vbkey",
            "fyh",
            "uts",
            "utafter",
        ]:
            if v in inputs:
                del inputs[v]

        d_sw.compute(**inputs)
        return self.slice_output(inputs)


class TranslateUbKE(TranslateFortranData2Py):
    def _call(self, *args, **kwargs):
        d_sw.ubke(
            *args,
            **kwargs,
            cosa=self.grid.cosa,
            rsina=self.grid.rsina,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute_buffer_2d(),
        )

    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = self._call
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "ut": {},
            "ub": grid.compute_dict_buffer_2d(),
        }
        self.in_vars["parameters"] = ["dt5", "dt4"]
        self.out_vars = {"ub": grid.compute_dict_buffer_2d()}


class TranslateVbKE(TranslateFortranData2Py):
    def _call(self, *args, **kwargs):
        d_sw.vbke(
            *args,
            **kwargs,
            cosa=self.grid.cosa,
            rsina=self.grid.rsina,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute_buffer_2d(),
        )

    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = self._call
        self.in_vars["data_vars"] = {
            "vc": {},
            "uc": {},
            "vt": {},
            "vb": grid.compute_dict_buffer_2d(),
        }
        self.in_vars["parameters"] = ["dt5", "dt4"]
        self.out_vars = {"vb": grid.compute_dict_buffer_2d()}
