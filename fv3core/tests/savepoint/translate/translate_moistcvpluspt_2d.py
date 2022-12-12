from gt4py.gtscript import PARALLEL, computation, interval

import pace.fv3core.stencils.moist_cv as moist_cv
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField
from pace.stencils.testing import TranslateFortranData2Py, pad_field_in_j


def moist_pt(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qice: FloatField,
    qgraupel: FloatField,
    q_con: FloatField,
    pt: FloatField,
    cappa: FloatField,
    delp: FloatField,
    delz: FloatField,
    r_vir: float,
):
    with computation(PARALLEL), interval(...):
        cvm, gz, q_con, cappa, pt = moist_cv.moist_pt_func(
            qvapor,
            qliquid,
            qrain,
            qsnow,
            qice,
            qgraupel,
            q_con,
            pt,
            cappa,
            delp,
            delz,
            r_vir,
        )


class MoistPT:
    """
    Class to test with DaCe orchestration. test class is MoistCVPlusPt_2d
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        grid,
    ):
        self._moist_cv_pt = stencil_factory.from_origin_domain(
            moist_pt,
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
        pt: FloatField,
        cappa: FloatField,
        delp: FloatField,
        delz: FloatField,
        r_vir: float,
    ):
        self._moist_cv_pt(
            qvapor,
            qliquid,
            qrain,
            qsnow,
            qice,
            qgraupel,
            q_con,
            pt,
            cappa,
            delp,
            delz,
            r_vir,
        )


class TranslateMoistCVPlusPt_2d(TranslateFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.stencil_factory = stencil_factory
        self.compute_func = MoistPT(stencil_factory, self.grid)  # type: ignore
        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "qvapor_js"},
            "qliquid": {"serialname": "qliquid_js"},
            "qice": {"serialname": "qice_js"},
            "qrain": {"serialname": "qrain_js"},
            "qsnow": {"serialname": "qsnow_js"},
            "qgraupel": {"serialname": "qgraupel_js"},
            "delp": {},
            "delz": {},
            "q_con": {},
            "pt": {},
            "cappa": {},
        }
        self.write_vars = ["gz", "cvm"]
        for k, v in self.in_vars["data_vars"].items():
            if k not in self.write_vars:
                v["axis"] = 1

        self.in_vars["parameters"] = ["r_vir"]
        self.out_vars = {
            "pt": {},
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
