import numpy as np

import pace.dsl.gt4py_utils as utils
from pace.physics.stencils.physics import atmos_phys_driver_statein
from pace.stencils.testing.translate_physics import TranslatePhysicsFortranData2Py
from pace.util.constants import KAPPA


class TranslateAtmosPhysDriverStatein(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "prsik": {"serialname": "IPD_prsik", "order": "F"},
            "phii": {"serialname": "IPD_phii", "order": "F"},
            "prsi": {"serialname": "IPD_prsi", "order": "F"},
            "delz": {"dycore": True},
            "delp": {"dycore": True},
            "qvapor": {"dycore": True},
            "qliquid": {"dycore": True},
            "qrain": {"dycore": True},
            "qice": {"dycore": True},
            "qsnow": {"dycore": True},
            "qgraupel": {"dycore": True},
            "qo3mr": {"dycore": True},
            "qcld": {"dycore": True},
            "pt": {"dycore": True, "order": "F"},
        }
        self.in_vars["parameters"] = []
        self.out_vars = {
            "prsik": self.in_vars["data_vars"]["prsik"],
            "prsi": self.in_vars["data_vars"]["prsi"],
            "phii": self.in_vars["data_vars"]["phii"],
            "pt": {
                "serialname": "IPD_tgrs",
                "out_roll_zero": True,
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "qgrs": {
                "serialname": "IPD_qgrs",
                "kend": namelist.npz,
                "order": "F",
                "manual": True,
            },
            "delp": {
                "serialname": "IPD_prsl",
                "out_roll_zero": True,
                "kend": namelist.npz - 1,
                "order": "F",
            },
        }
        self.stencil_factory = stencil_factory
        self.compute_func = self.stencil_factory.from_origin_domain(
            atmos_phys_driver_statein,
            origin=self.stencil_factory.grid_indexing.origin_compute(),
            domain=self.stencil_factory.grid_indexing.domain_compute(add=(0, 0, 1)),
            externals={
                "nwat": 6,
                "ptop": 300,  # hard coded before we can call
                # ak from grid: state["ak"][0]
                "pk0inv": (1.0 / 1.0e5) ** KAPPA,
                "pktop": (300.0 / 1.0e5) ** KAPPA,
            },
        )

    def post_process_qgrs(self, inputs):
        qgrs = np.stack(
            (
                inputs["qvapor"],
                inputs["qliquid"],
                inputs["qrain"],
                inputs["qice"],
                inputs["qsnow"],
                inputs["qgraupel"],
                inputs["qo3mr"],
                inputs["qsgs_tke"],
                inputs["qcld"],
            ),
            axis=-1,
        )
        info = self.out_vars["qgrs"]
        self.update_info(info, inputs)
        ds = self.grid.compute_dict()
        ds.update(info)
        k_length = info["kend"] if "kend" in info else self.namelist.npz
        index_order = info["order"] if "order" in info else "C"
        ij_slice = self.grid.slice_dict(ds)
        qgrs = qgrs[ij_slice[0], ij_slice[1], 0:k_length, :]
        qgrs = np.reshape(
            qgrs,
            (qgrs.shape[0] * qgrs.shape[1], qgrs.shape[2], qgrs.shape[3]),
            order=index_order,
        )

        return qgrs[:, ::-1, :]

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        qsgs_tke = utils.make_storage_from_shape(
            self.maxshape,
            origin=(0, 0, 0),
            backend=self.stencil_factory.backend,
        )
        dm = utils.make_storage_from_shape(
            self.maxshape,
            origin=(0, 0, 0),
            backend=self.stencil_factory.backend,
        )
        inputs["qsgs_tke"] = qsgs_tke
        inputs["dm"] = dm
        self.compute_func(**inputs)
        out = self.slice_output(inputs)
        out["IPD_qgrs"] = self.post_process_qgrs(inputs)
        return out
