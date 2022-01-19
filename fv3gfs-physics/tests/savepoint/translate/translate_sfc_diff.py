from fv3gfs.physics.stencils.surface import sfc_diff
from pace.stencils.testing.translate_physics import TranslatePhysicsFortranData2Py


class TranslateSfcDiff1(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "t1": {"serialname": "tgrs_sfc", "sfc2d": True},
            "q1": {"serialname": "qgrs_sfc", "sfc2d": True},
            "z1": {"serialname": "zlvl", "sfc2d": True},
            "wind": {"serialname": "wind", "sfc2d": True},
            "prslki": {"serialname": "work3", "sfc2d": True},
            "sigmaf": {"serialname": "sigmaf", "sfc2d": True},
            "vegtype": {"serialname": "vegtype", "sfc2d": True},
            "shdmax": {"serialname": "shdmax", "sfc2d": True},
            "z0pert": {"serialname": "z01d", "sfc2d": True},
            "ztpert": {"serialname": "zt1d", "sfc2d": True},
            "flag_iter": {"serialname": "flag_iter", "sfc2d": True},
            "u10m": {"serialname": "u10m", "sfc2d": True},
            "v10m": {"serialname": "v10m", "sfc2d": True},
            "wet": {"serialname": "wet", "sfc2d": True},
            "dry": {"serialname": "dry", "sfc2d": True},
            "icy": {"serialname": "icy", "sfc2d": True},
            "tskin": {"serialname": "tsfc3", "sfc2d": True},
            "tsurf": {"serialname": "tsurf3", "sfc2d": True},
            "snwdph": {"serialname": "snowd3", "sfc2d": True},
            "z0rl": {"serialname": "zorl3", "sfc2d": True},
            "ustar": {"serialname": "uustar3", "sfc2d": True},
            "cm": {
                "serialname": "cd3",
                "sfc2d": True,
            },
            "ch": {"serialname": "cdq3", "sfc2d": True},
            "rb": {"serialname": "rb3", "sfc2d": True},
            "stress": {"serialname": "stress3", "sfc2d": True},
            "fm": {"serialname": "ffmm3", "sfc2d": True},
            "fh": {"serialname": "ffhh3", "sfc2d": True},
            "fm10": {"serialname": "fm103", "sfc2d": True},
            "fh2": {"serialname": "fh23", "sfc2d": True},
        }
        self.in_vars["parameters"] = ["ivegsrc", "redrag", "sfc_z0_type"]
        self.out_vars = {
            "cm": self.in_vars["data_vars"]["cm"],
            "ch": self.in_vars["data_vars"]["ch"],
            "rb": self.in_vars["data_vars"]["rb"],
            "fm": self.in_vars["data_vars"]["fm"],
            "fh": self.in_vars["data_vars"]["fh"],
            "fm10": self.in_vars["data_vars"]["fm10"],
            "fh2": self.in_vars["data_vars"]["fh2"],
        }
        self.compute_func = stencil_factory.from_origin_domain(
            sfc_diff,
            origin=stencil_factory.grid_indexing.origin_full(),
            domain=stencil_factory.grid_indexing.domain_full(add=(0, 0, 1)),
        )

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        self.compute_func(**inputs)

        return self.slice_surface2d_output(inputs)


class TranslateSfcDiff2(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "t1": {"serialname": "tgrs_sfc", "sfc2d": True},
            "q1": {"serialname": "qgrs_sfc", "sfc2d": True},
            "z1": {"serialname": "zlvl", "sfc2d": True},
            "wind": {"serialname": "wind", "sfc2d": True},
            "prslki": {"serialname": "work3", "sfc2d": True},
            "sigmaf": {"serialname": "sigmaf", "sfc2d": True},
            "vegtype": {"serialname": "vegtype", "sfc2d": True},
            "shdmax": {"serialname": "shdmax", "sfc2d": True},
            "z0pert": {"serialname": "z01d", "sfc2d": True},
            "ztpert": {"serialname": "zt1d", "sfc2d": True},
            "flag_iter": {"serialname": "flag_iter", "sfc2d": True},
            "u10m": {"serialname": "u10m", "sfc2d": True},
            "v10m": {"serialname": "v10m", "sfc2d": True},
            "wet": {"serialname": "wet", "sfc2d": True},
            "dry": {"serialname": "dry", "sfc2d": True},
            "icy": {"serialname": "icy", "sfc2d": True},
            "tskin": {"serialname": "tsfc3", "sfc2d": True},
            "tsurf": {"serialname": "tsurf3", "sfc2d": True},
            "snwdph": {"serialname": "snowd3", "sfc2d": True},
            "z0rl": {"serialname": "zorl3", "sfc2d": True},
            "ustar": {"serialname": "uustar3", "sfc2d": True},
            "cm": {
                "serialname": "cd3",
                "sfc2d": True,
            },
            "ch": {"serialname": "cdq3", "sfc2d": True},
            "rb": {"serialname": "rb3", "sfc2d": True},
            "stress": {"serialname": "stress3", "sfc2d": True},
            "fm": {"serialname": "ffmm3", "sfc2d": True},
            "fh": {"serialname": "ffhh3", "sfc2d": True},
            "fm10": {"serialname": "fm103", "sfc2d": True},
            "fh2": {"serialname": "fh23", "sfc2d": True},
        }
        self.in_vars["parameters"] = ["ivegsrc", "redrag", "sfc_z0_type"]
        self.out_vars = {
            "cm": self.in_vars["data_vars"]["cm"],
            "ch": self.in_vars["data_vars"]["ch"],
            "rb": self.in_vars["data_vars"]["rb"],
            "fm": self.in_vars["data_vars"]["fm"],
            "fh": self.in_vars["data_vars"]["fh"],
            "fm10": self.in_vars["data_vars"]["fm10"],
            "fh2": self.in_vars["data_vars"]["fh2"],
        }
        self.compute_func = stencil_factory.from_origin_domain(
            sfc_diff,
            origin=stencil_factory.grid_indexing.origin_full(),
            domain=stencil_factory.grid_indexing.domain_full(add=(0, 0, 1)),
        )

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        self.compute_func(**inputs)
        return self.slice_surface2d_output(inputs)
