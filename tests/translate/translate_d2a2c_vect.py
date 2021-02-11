from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.d2a2c_vect as d2a2c_vect
from fv3core.decorators import gtstencil
from fv3core.testing import TranslateFortranData2Py
from fv3core.utils.typing import FloatField


def test_d2a2c_vect(
    cosa_s: FloatField,
    cosa_u: FloatField,
    cosa_v: FloatField,
    dxa: FloatField,
    dya: FloatField,
    rsin2: FloatField,
    rsin_u: FloatField,
    rsin_v: FloatField,
    sin_sg1: FloatField,
    sin_sg2: FloatField,
    sin_sg3: FloatField,
    sin_sg4: FloatField,
    u: FloatField,
    ua: FloatField,
    uc: FloatField,
    utc: FloatField,
    v: FloatField,
    va: FloatField,
    vc: FloatField,
    vtc: FloatField,
):
    with computation(PARALLEL), interval(...):
        uc, vc, ua, va, utc, vtc = d2a2c_vect.d2a2c_vect(
            cosa_s,
            cosa_u,
            cosa_v,
            dxa,
            dya,
            rsin2,
            rsin_u,
            rsin_v,
            sin_sg1,
            sin_sg2,
            sin_sg3,
            sin_sg4,
            u,
            ua,
            uc,
            utc,
            v,
            va,
            vc,
            vtc,
        )


class TranslateD2A2C_Vect(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "u": {},
            "v": {},
            "ua": {},
            "va": {},
            "utc": {},
            "vtc": {},
        }
        self.in_vars["parameters"] = ["dord4"]
        self.out_vars = {
            "uc": grid.x3d_domain_dict(),
            "vc": grid.y3d_domain_dict(),
            "ua": {},
            "va": {},
            "utc": {},
            "vtc": {},
        }
        # TODO: This seems to be needed primarily for the edge_interpolate_4
        # methods, can we rejigger the order of operations to make it match to
        # more precision?
        self.max_error = 2e-10

    def compute(self, inputs):
        if spec.namelist.npx != spec.namelist.npy:
            raise NotImplementedError("D2A2C assumes a square grid")
        if spec.namelist.npx <= 13 and spec.namelist.layout[0] > 1:
            D2A2C_AVG_OFFSET = -1
        else:
            D2A2C_AVG_OFFSET = 3

        stencil = gtstencil(
            definition=test_d2a2c_vect,
            externals={"D2A2C_AVG_OFFSET": D2A2C_AVG_OFFSET},
        )
        self.make_storage_data_input_vars(inputs)
        assert bool(inputs["dord4"]) is True
        del inputs["dord4"]
        stencil(
            cosa_s=self.grid.cosa_s,
            cosa_u=self.grid.cosa_u,
            cosa_v=self.grid.cosa_v,
            dxa=self.grid.dxa,
            dya=self.grid.dya,
            rsin2=self.grid.rsin2,
            rsin_u=self.grid.rsin_u,
            rsin_v=self.grid.rsin_v,
            sin_sg1=self.grid.sin_sg1,
            sin_sg2=self.grid.sin_sg2,
            sin_sg3=self.grid.sin_sg3,
            sin_sg4=self.grid.sin_sg4,
            **inputs,
            origin=self.grid.compute_origin(add=(-2, -2, 0)),
            domain=self.grid.domain_shape_compute(add=(4, 4, 0)),
        )
        return self.slice_output(inputs)
