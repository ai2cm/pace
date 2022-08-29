import pace.fv3core.stencils.moist_cv as moist_cv
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField
from pace.fv3core.testing import TranslateDycoreFortranData2Py
from pace.stencils.testing import pad_field_in_j
from pace.util import Namelist


class MoistPKZ:
    """
    Class to test with DaCe orchestration. test class is MoistCVPlusPkz_2d
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        grid,
    ):
        self._moist_cv_pkz = stencil_factory.from_origin_domain(
            moist_cv.moist_pkz,
            origin=grid.compute_origin(),
            domain=(grid.nic, 1, grid.npz),
        )

    def __call__(
        self,
        qvapor: FloatField,
        qliquid: FloatField,
        qrain: FloatField,
        qsnow: FloatField,
        qice: FloatField,
        qgraupel: FloatField,
        q_con: FloatField,
        gz: FloatField,
        cvm: FloatField,
        pkz: FloatField,
        pt: FloatField,
        cappa: FloatField,
        delp: FloatField,
        delz: FloatField,
        r_vir: float,
    ):

        self._moist_cv_pkz(
            qvapor,
            qliquid,
            qrain,
            qsnow,
            qice,
            qgraupel,
            q_con,
            gz,
            cvm,
            pkz,
            pt,
            cappa,
            delp,
            delz,
            r_vir,
        )


class TranslateMoistCVPlusPkz_2d(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.stencil_factory = stencil_factory
        self.compute_func = MoistPKZ(stencil_factory, self.grid)

        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "qvapor_js"},
            "qliquid": {"serialname": "qliquid_js"},
            "qice": {"serialname": "qice_js"},
            "qrain": {"serialname": "qrain_js"},
            "qsnow": {"serialname": "qsnow_js"},
            "qgraupel": {"serialname": "qgraupel_js"},
            "gz": {"serialname": "gz1d", "kstart": grid.is_, "axis": 0},
            "cvm": {"kstart": grid.is_, "axis": 0},
            "delp": {},
            "delz": {},
            "q_con": {},
            "pkz": {"istart": grid.is_, "jstart": grid.js},
            "pt": {},
            "cappa": {},
        }
        self.write_vars = ["gz", "cvm"]
        for k, v in self.in_vars["data_vars"].items():
            if k not in self.write_vars:
                v["axis"] = 1

        self.in_vars["parameters"] = ["r_vir"]
        self.out_vars = {
            "gz": {
                "serialname": "gz1d",
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.js,
                "kstart": grid.npz - 1,
                "kend": grid.npz - 1,
            },
            "cvm": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.js,
                "kstart": grid.npz - 1,
                "kend": grid.npz - 1,
            },
            "pkz": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
            "cappa": {},
            "q_con": {},
        }

    def compute_from_storage(self, inputs):
        for name, value in inputs.items():
            if hasattr(value, "shape") and len(value.shape) > 1 and value.shape[1] == 1:
                inputs[name] = self.make_storage_data(
                    pad_field_in_j(
                        value, self.grid.njd, backend=self.stencil_factory.backend
                    )
                )
        self.compute_func(**inputs)
        return inputs
