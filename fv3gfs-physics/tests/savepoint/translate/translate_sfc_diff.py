from fv3gfs.physics.stencils.surface import sfc_diff
from pace.stencils.testing.translate_physics import TranslatePhysicsFortranData2Py


class TranslateSfcDiff1(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "t1": {"serialname": "tgrs_sfc"},
            "q1": {"serialname": "qgrs_sfc"},
            "z1": {"serialname": "zlvl"},
            "wind": {"serialname": "wind"},
            "prslki": {"serialname": "work3"},
            "sigmaf": {"serialname": "sigmaf"},
            "vegtype": {"serialname": "vegtype"},
            "shdmax": {"serialname": "shdmax"},
            "ivegsrc": {"serialname": "ivegsrc"},
            "z0pert": {"serialname": "z01d"},
            "ztpert": {"serialname": "zt1d"},
            "flag_iter": {"serialname": "flag_iter"},
            "redrag": {"serialname": "redrag"},
            "u10m": {"serialname": "u10m"},
            "v10m": {"serialname": "v10m"},
            "sfc_z0_type": {"serialname": "sfc_z0_type"},
            "wet": {"serialname": "wet"},
            "dry": {"serialname": "dry"},
            "icy": {"serialname": "icy"},
            "tskin": {"serialname": "tsfc3"},
            "tsurf": {"serialname": "tsurf3"},
            "snwdph": {"serialname": "snowd3"},
            "z0rl": {"serialname": "zorl3"},
            "ustar": {"serialname": "uustar3"},
            "cm": {"serialname": "cd3"},
            "ch": {"serialname": "cdq3"},
            "rb": {"serialname": "rb3"},
            "stress": {"serialname": "stress3"},
            "fm": {"serialname": "ffmm3"},
            "fh": {"serialname": "ffhh3"},
            "fm10": {"serialname": "fm103"},
            "fh2": {"serialname": "fh23"},
        }
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
        output = {}
        
        return self.slice_surface2d_output(inputs)


class TranslateSfcDiff2(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "t1": {"serialname": "tgrs_sfc"},
            "q1": {"serialname": "qgrs_sfc"},
            "z1": {"serialname": "zlvl"},
            "wind": {"serialname": "wind"},
            "prslki": {"serialname": "work3"},
            "sigmaf": {"serialname": "sigmaf"},
            "vegtype": {"serialname": "vegtype"},
            "shdmax": {"serialname": "shdmax"},
            "ivegsrc": {"serialname": "ivegsrc"},
            "z0pert": {"serialname": "z01d"},
            "ztpert": {"serialname": "zt1d"},
            "flag_iter": {"serialname": "flag_iter"},
            "redrag": {"serialname": "redrag"},
            "u10m": {"serialname": "u10m"},
            "v10m": {"serialname": "v10m"},
            "sfc_z0_type": {"serialname": "sfc_z0_type"},
            "wet": {"serialname": "wet"},
            "dry": {"serialname": "dry"},
            "icy": {"serialname": "icy"},
            "tskin": {"serialname": "tsfc3"},
            "tsurf": {"serialname": "tsurf3"},
            "snwdph": {"serialname": "snowd3"},
            "z0rl": {"serialname": "zorl3"},
            "ustar": {"serialname": "uustar3"},
            "cm": {"serialname": "cd3"},
            "ch": {"serialname": "cdq3"},
            "rb": {"serialname": "rb3"},
            "stress": {"serialname": "stress3"},
            "fm": {"serialname": "ffmm3"},
            "fh": {"serialname": "ffhh3"},
            "fm10": {"serialname": "fm103"},
            "fh2": {"serialname": "fh23"},
        }
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
