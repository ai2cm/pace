from gt4py.gtscript import PARALLEL, computation, interval

import fv3core.stencils.c_sw as c_sw
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.c_sw import (
    circulation_cgrid,
    divergence_corner,
    transportdelp,
    update_vorticity_and_kinetic_energy,
    vorticitytransport,
)
from fv3core.testing import TranslateFortranData2Py
from fv3core.utils.typing import FloatField


class TranslateC_SW(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = c_sw.compute
        self.in_vars["data_vars"] = {
            "delp": {},
            "pt": {},
            "u": {"jend": grid.jed + 1},
            "v": {"iend": grid.ied + 1},
            "w": {},
            "uc": {"iend": grid.ied + 1},
            "vc": {"jend": grid.jed + 1},
            "ua": {},
            "va": {},
            "ut": {},
            "vt": {},
            "omga": {"serialname": "omgad"},
            "divgd": {"iend": grid.ied + 1, "jend": grid.jed + 1},
        }
        self.in_vars["parameters"] = ["dt2"]
        for name, info in self.in_vars["data_vars"].items():
            info["serialname"] = name + "d"
        self.out_vars = {}
        for v, d in self.in_vars["data_vars"].items():
            if v not in ("delp", "pt", "w"):
                self.out_vars[v] = d
        for servar in ["delpcd", "ptcd"]:
            self.out_vars[servar] = {}
        # TODO: Fix edge_interpolate4 in d2a2c_vect to match closer and the
        # variables here should as well.
        self.max_error = 2e-10

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        delpc, ptc = self.compute_func(**inputs)
        return self.slice_output(inputs, {"delpcd": delpc, "ptcd": ptc})


@gtstencil()
def transportdelp_stencil(
    delp: FloatField,
    pt: FloatField,
    utc: FloatField,
    vtc: FloatField,
    w: FloatField,
    rarea: FloatField,
    delpc: FloatField,
    ptc: FloatField,
    wc: FloatField,
):
    with computation(PARALLEL), interval(...):
        delpc, ptc, wc = transportdelp(delp, pt, utc, vtc, w, rarea)


class TranslateTransportDelp(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "delp": {},
            "pt": {},
            "utc": {},
            "vtc": {},
            "w": {},
            "wc": {},
        }
        self.out_vars = {"delpc": {}, "ptc": {}, "wc": {}}

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        orig = (self.grid.is_ - 1, self.grid.js - 1, 0)
        inputs["delpc"] = utils.make_storage_from_shape(
            inputs["delp"].shape, origin=orig
        )
        inputs["ptc"] = utils.make_storage_from_shape(inputs["pt"].shape, origin=orig)
        transportdelp_stencil(
            **inputs,
            rarea=self.grid.rarea,
            origin=orig,
            domain=self.grid.domain_shape_compute(add=(2, 2, 0)),
        )
        return self.slice_output(inputs)


@gtstencil()
def divergence_corner_stencil(
    u: FloatField,
    v: FloatField,
    ua: FloatField,
    va: FloatField,
    dxc: FloatField,
    dyc: FloatField,
    sin_sg1: FloatField,
    sin_sg2: FloatField,
    sin_sg3: FloatField,
    sin_sg4: FloatField,
    cos_sg1: FloatField,
    cos_sg2: FloatField,
    cos_sg3: FloatField,
    cos_sg4: FloatField,
    rarea_c: FloatField,
    divg_d: FloatField,
):
    with computation(PARALLEL), interval(...):
        divg_d = divergence_corner(
            u,
            v,
            ua,
            va,
            dxc,
            dyc,
            sin_sg1,
            sin_sg2,
            sin_sg3,
            sin_sg4,
            cos_sg1,
            cos_sg2,
            cos_sg3,
            cos_sg4,
            rarea_c,
        )


class TranslateDivergenceCorner(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "u": {
                "istart": grid.isd,
                "iend": grid.ied,
                "jstart": grid.jsd,
                "jend": grid.jed + 1,
            },
            "v": {
                "istart": grid.isd,
                "iend": grid.ied + 1,
                "jstart": grid.jsd,
                "jend": grid.jed,
            },
            "ua": {},
            "va": {},
            "divg_d": {},
        }
        self.out_vars = {
            "divg_d": {
                "istart": grid.isd,
                "iend": grid.ied + 1,
                "jstart": grid.jsd,
                "jend": grid.jed + 1,
            }
        }

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        divergence_corner_stencil(
            **inputs,
            dxc=self.grid.dxc,
            dyc=self.grid.dyc,
            sin_sg1=self.grid.sin_sg1,
            sin_sg2=self.grid.sin_sg2,
            sin_sg3=self.grid.sin_sg3,
            sin_sg4=self.grid.sin_sg4,
            cos_sg1=self.grid.cos_sg1,
            cos_sg2=self.grid.cos_sg2,
            cos_sg3=self.grid.cos_sg3,
            cos_sg4=self.grid.cos_sg4,
            rarea_c=self.grid.rarea_c,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(add=(1, 1, 0)),
        )
        return self.slice_output({"divg_d": inputs["divg_d"]})


@gtstencil()
def circulation_cgrid_stencil(
    uc: FloatField, vc: FloatField, dxc: FloatField, dyc: FloatField, vort_c: FloatField
):
    with computation(PARALLEL), interval(...):
        vort_c = circulation_cgrid(uc, vc, dxc, dyc)


class TranslateCirculation_Cgrid(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "vort_c": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
            },
        }
        self.out_vars = {
            "vort_c": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
            }
        }

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        circulation_cgrid_stencil(
            **inputs,
            dxc=self.grid.dxc,
            dyc=self.grid.dyc,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(add=(1, 1, 0)),
        )
        return self.slice_output({"vort_c": inputs["vort_c"]})


@gtstencil()
def update_vorticity_and_kinetic_energy_stencil(
    ua: FloatField,
    va: FloatField,
    uc: FloatField,
    vc: FloatField,
    u: FloatField,
    v: FloatField,
    sin_sg1: FloatField,
    cos_sg1: FloatField,
    sin_sg2: FloatField,
    cos_sg2: FloatField,
    sin_sg3: FloatField,
    cos_sg3: FloatField,
    sin_sg4: FloatField,
    cos_sg4: FloatField,
    ke_c: FloatField,
    vort_c: FloatField,
    dt2: float,
):
    with computation(PARALLEL), interval(...):
        ke_c, vort_c = update_vorticity_and_kinetic_energy(
            ua,
            va,
            uc,
            vc,
            u,
            v,
            sin_sg1,
            cos_sg1,
            sin_sg2,
            cos_sg2,
            sin_sg3,
            cos_sg3,
            sin_sg4,
            cos_sg4,
            dt2,
        )


class TranslateKE_C_SW(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "u": {},
            "v": {},
            "ua": {},
            "va": {},
        }
        self.in_vars["parameters"] = ["dt2"]
        self.out_vars = {
            "ke_c": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
            },
            "vort_c": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
            },
        }

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        inputs["ke_c"] = utils.make_storage_from_shape(inputs["uc"].shape)
        inputs["vort_c"] = utils.make_storage_from_shape(inputs["uc"].shape)
        update_vorticity_and_kinetic_energy_stencil(
            sin_sg1=self.grid.sin_sg1,
            cos_sg1=self.grid.cos_sg1,
            sin_sg2=self.grid.sin_sg2,
            cos_sg2=self.grid.cos_sg2,
            sin_sg3=self.grid.sin_sg3,
            cos_sg3=self.grid.cos_sg3,
            sin_sg4=self.grid.sin_sg4,
            cos_sg4=self.grid.cos_sg4,
            **inputs,
            origin=self.grid.compute_origin(add=(-1, -1, 0)),
            domain=self.grid.domain_shape_compute(add=(2, 2, 0)),
        )
        return self.slice_output(inputs)


@gtstencil()
def vorticitytransport_stencil(
    vort_c: FloatField,
    ke_c: FloatField,
    u: FloatField,
    v: FloatField,
    uc: FloatField,
    vc: FloatField,
    cosa_u: FloatField,
    sina_u: FloatField,
    cosa_v: FloatField,
    sina_v: FloatField,
    rdxc: FloatField,
    rdyc: FloatField,
    dt2: float,
):
    with computation(PARALLEL), interval(...):
        uc, vc = vorticitytransport(
            vort_c, ke_c, u, v, uc, vc, cosa_u, sina_u, cosa_v, sina_v, rdxc, rdyc, dt2
        )


class TranslateVorticityTransport_Cgrid(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "vort_c": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
            },
            "ke_c": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
            },
            "u": {},
            "v": {},
        }
        self.in_vars["parameters"] = ["dt2"]
        self.out_vars = {"uc": grid.x3d_domain_dict(), "vc": grid.y3d_domain_dict()}

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        vorticitytransport_stencil(
            **inputs,
            cosa_u=self.grid.cosa_u,
            sina_u=self.grid.sina_u,
            cosa_v=self.grid.cosa_v,
            sina_v=self.grid.sina_v,
            rdxc=self.grid.rdxc,
            rdyc=self.grid.rdyc,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(add=(1, 1, 0)),
        )
        return self.slice_output(inputs)
