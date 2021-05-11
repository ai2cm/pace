from gt4py.gtscript import BACKWARD, FORWARD, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.global_constants as constants
from fv3core.decorators import FrozenStencil, gtstencil
from fv3core.stencils.basic_operations import adjust_divide_stencil
from fv3core.stencils.moist_cv import moist_pt_last_step
from fv3core.stencils.saturation_adjustment import SatAdjust3d
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK


def copy_from_below(a: FloatField, b: FloatField):
    with computation(PARALLEL), interval(1, None):
        b = a[0, 0, -1]


@gtstencil
def init_phis(hs: FloatField, delz: FloatField, phis: FloatField, te_2d: FloatFieldIJ):
    with computation(BACKWARD):
        with interval(-1, None):
            te_2d = 0.0
            phis = hs
        with interval(0, -1):
            te_2d = 0.0
            phis = phis[0, 0, 1] - constants.GRAV * delz


def sum_te(te: FloatField, te0_2d: FloatField):
    with computation(FORWARD):
        with interval(0, None):
            te0_2d = te0_2d[0, 0, -1] + te


class VerticalRemapping2:
    def __init__(self, pfull):
        self.grid = spec.grid
        self.namelist = spec.namelist

        self._copy_from_below_stencil = FrozenStencil(
            copy_from_below,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(),
        )

        self._moist_cv_last_step_stencil = FrozenStencil(
            moist_pt_last_step,
            origin=(self.grid.is_, self.grid.js, 0),
            domain=(self.grid.nic, self.grid.njc, self.grid.npz + 1),
        )

        self._basic_adjust_divide_stencil = FrozenStencil(
            adjust_divide_stencil,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(),
        )

        self._do_sat_adjust = self.namelist.do_sat_adj

        self.kmp = self.grid.npz - 1
        for k in range(pfull.shape[0]):
            if pfull[k] > 10.0e2:
                self.kmp = k
                break

        self._saturation_adjustment = SatAdjust3d(self.kmp)

        self._sum_te_stencil = FrozenStencil(
            sum_te,
            origin=(self.grid.is_, self.grid.js, self.kmp),
            domain=(self.grid.nic, self.grid.njc, self.grid.npz - self.kmp),
        )

    def __call__(
        self,
        qvapor: FloatField,
        qliquid: FloatField,
        qice: FloatField,
        qrain: FloatField,
        qsnow: FloatField,
        qgraupel: FloatField,
        qcld: FloatField,
        pt: FloatField,
        delp: FloatField,
        delz: FloatField,
        peln: FloatField,
        u: FloatField,
        v: FloatField,
        w: FloatField,
        ua: FloatField,
        cappa: FloatField,
        q_con: FloatField,
        gz: FloatField,
        pkz: FloatField,
        pk: FloatField,
        pe: FloatField,
        hs: FloatFieldIJ,
        te0_2d: FloatFieldIJ,
        te: FloatField,
        cvm: FloatField,
        pfull: FloatFieldK,
        ptop: float,
        akap: float,
        r_vir: float,
        last_step: bool,
        pdt: float,
        mdt: float,
        consv: float,
        do_adiabatic_init: bool,
    ):
        self._copy_from_below_stencil(ua, pe)
        dtmp = 0.0
        if last_step and not do_adiabatic_init:
            if consv > constants.CONSV_MIN:
                raise NotImplementedError(
                    "We do not support consv_te > 0.001 "
                    "because that would trigger an allReduce"
                )
            elif consv < -constants.CONSV_MIN:
                raise Exception(
                    "Unimplemented/untested case consv("
                    + str(consv)
                    + ")  < -CONSV_MIN("
                    + str(-constants.CONSV_MIN)
                    + ")"
                )

        if self._do_sat_adjust:
            fast_mp_consv = not do_adiabatic_init and consv > constants.CONSV_MIN
            self._saturation_adjustment(
                te,
                qvapor,
                qliquid,
                qice,
                qrain,
                qsnow,
                qgraupel,
                qcld,
                hs,
                peln,
                delp,
                delz,
                q_con,
                pt,
                pkz,
                cappa,
                r_vir,
                mdt,
                fast_mp_consv,
                last_step,
                akap,
                self.kmp,
            )
            if fast_mp_consv:
                self._sum_te_stencil(
                    te,
                    te0_2d,
                )
        if last_step:
            self._moist_cv_last_step_stencil(
                qvapor,
                qliquid,
                qrain,
                qsnow,
                qice,
                qgraupel,
                gz,
                pt,
                pkz,
                dtmp,
                r_vir,
            )
        else:
            self._basic_adjust_divide_stencil(pkz, pt)
