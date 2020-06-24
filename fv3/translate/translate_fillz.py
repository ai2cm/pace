from .translate import TranslateFortranData2Py
import fv3.stencils.fillz as Fillz
import numpy as np


class TranslateFillz(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = Fillz.compute
        self.in_vars["data_vars"] = {
            "dp2": {"istart": grid.is_, "iend": grid.ie},
            "qvapor": {"serialname": "q2vapor_js", "istart": grid.is_, "iend": grid.ie},
            "qliquid": {
                "serialname": "q2liquid_js",
                "istart": grid.is_,
                "iend": grid.ie,
            },
            "qice": {"serialname": "q2ice_js", "istart": grid.is_, "iend": grid.ie},
            "qrain": {"serialname": "q2rain_js", "istart": grid.is_, "iend": grid.ie},
            "qsnow": {"serialname": "q2snow_js", "istart": grid.is_, "iend": grid.ie},
            "qgraupel": {
                "serialname": "q2graupel_js",
                "istart": grid.is_,
                "iend": grid.ie,
            },
            "qcld": {"serialname": "q2cld_js", "istart": grid.is_, "iend": grid.ie},
        }
        self.in_vars["parameters"] = ["im", "km", "nq"]
        self.out_vars = {
            "qvapor": {"serialname": "q2vapor_js", "istart": grid.is_, "iend": grid.ie},
            "qliquid": {
                "serialname": "q2liquid_js",
                "istart": grid.is_,
                "iend": grid.ie,
            },
            "qice": {"serialname": "q2ice_js", "istart": grid.is_, "iend": grid.ie},
            "qrain": {"serialname": "q2rain_js", "istart": grid.is_, "iend": grid.ie},
            "qsnow": {"serialname": "q2snow_js", "istart": grid.is_, "iend": grid.ie},
            "qgraupel": {
                "serialname": "q2graupel_js",
                "istart": grid.is_,
                "iend": grid.ie,
            },
            "qcld": {"serialname": "q2cld_js", "istart": grid.is_, "iend": grid.ie},
        }
        self.max_error = 1e-13

    def make_storage_data_input_vars(self, inputs, storage_vars=None):
        if storage_vars is None:
            storage_vars = self.storage_vars()
        for p in self.in_vars["parameters"]:
            if type(inputs[p]) in [np.int64, np.int32]:
                inputs[p] = int(inputs[p])
        for d, info in storage_vars.items():
            serialname = info["serialname"] if "serialname" in info else d
            self.update_info(info, inputs)
            if "kaxis" in info:
                inputs[serialname] = np.moveaxis(inputs[serialname], info["kaxis"], 2)
            istart, jstart, kstart = self.collect_start_indices(
                inputs[serialname].shape, info
            )

            shapes = np.squeeze(inputs[serialname]).shape
            if len(shapes) == 2:
                # suppress j
                dummy_axes = [1]
            elif len(shapes) == 1:
                # suppress j and k
                dummy_axes = [1, 2]
            else:
                dummy_axes = None

            inputs[d] = self.make_storage_data(
                np.squeeze(inputs[serialname]),
                istart=istart,
                jstart=jstart,
                kstart=kstart,
                dummy_axes=dummy_axes,
            )
            if d != serialname:
                del inputs[serialname]

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        inputs["jslice"] = slice(0, 1)
        qvapor, qliquid, qice, qrain, qsnow, qgraupel, qcld = self.compute_func(
            **inputs
        )
        return self.slice_output(
            inputs,
            {
                "qvapor_js": qvapor,
                "qliquid_js": qliquid,
                "qice_js": qice,
                "qrain_js": qrain,
                "qsnow_js": qsnow,
                "qgraupel_js": qgraupel,
                "qcld_js": qcld,
            },
        )
