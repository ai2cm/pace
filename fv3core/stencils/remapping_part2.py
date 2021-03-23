from gt4py.gtscript import BACKWARD, FORWARD, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.basic_operations as basic
import fv3core.stencils.moist_cv as moist_cv
import fv3core.stencils.saturation_adjustment as saturation_adjustment
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK


@gtstencil()
def copy_from_below(a: FloatField, b: FloatField):
    with computation(PARALLEL), interval(1, None):
        b = a[0, 0, -1]


@gtstencil()
def init_phis(hs: FloatField, delz: FloatField, phis: FloatField, te_2d: FloatFieldIJ):
    with computation(BACKWARD):
        with interval(-1, None):
            te_2d = 0.0
            phis = hs
        with interval(0, -1):
            te_2d = 0.0
            phis = phis[0, 0, 1] - constants.GRAV * delz


@gtstencil()
def sum_z1(
    pkz: FloatField,
    delp: FloatField,
    te0_2d: FloatField,
    te_2d: FloatFieldIJ,
    zsum1: FloatFieldIJ,
):
    with computation(FORWARD):
        with interval(0, 1):
            te_2d = te0_2d - te_2d
            zsum1 = pkz * delp
        with interval(1, None):
            te_2d = te0_2d - te_2d
            zsum1 = zsum1[0, 0, -1] + pkz * delp


@gtstencil()
def sum_te(te: FloatField, te0_2d: FloatField):
    with computation(FORWARD):
        with interval(0, None):
            te0_2d = te0_2d[0, 0, -1] + te


def compute(
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
    grid = spec.grid
    copy_from_below(
        ua, pe, origin=grid.compute_origin(), domain=grid.domain_shape_compute()
    )
    dtmp = 0.0
    phis = utils.make_storage_from_shape(pt.shape, grid.compute_origin())
    te_2d = utils.make_storage_from_shape(pt.shape[0:2], grid.compute_origin())
    zsum1 = utils.make_storage_from_shape(pt.shape[0:2], grid.compute_origin())
    if spec.namelist.do_sat_adj:
        fast_mp_consv = not do_adiabatic_init and consv > constants.CONSV_MIN
        # TODO pfull is a 1d var
        kmp = grid.npz - 1
        for k in range(pfull.shape[0]):
            if pfull[k] > 10.0e2:
                kmp = k
                break
    if last_step and not do_adiabatic_init:
        if consv > constants.CONSV_MIN:
            if spec.namelist.hydrostatic:
                raise Exception("Hydrostatic not supported")
            else:
                init_phis(
                    hs,
                    delz,
                    phis,
                    te_2d,
                    origin=grid.compute_origin(),
                    domain=(grid.nic, grid.njc, grid.npz + 1),
                )
                moist_cv.compute_te(
                    qvapor,
                    qliquid,
                    qice,
                    qrain,
                    qsnow,
                    qgraupel,
                    te_2d,
                    gz,
                    cvm,
                    delp,
                    q_con,
                    pt,
                    phis,
                    w,
                    u,
                    v,
                    r_vir,
                )
            sum_z1(
                pkz,
                delp,
                te0_2d,
                te_2d,
                zsum1,
                origin=grid.compute_origin(),
                domain=(grid.nic, grid.njc, grid.npz),
            )
            # dtmp = consv * g_sum(te_2d, grid.area_64)  # global mpi step
            # dtmp = dtmp / (constants.CV_AIR * g_sum(zsum1, grid.area_64))
            raise NotImplementedError(
                "We do not support consv_te > 0.001 "
                "because that would trigger an allReduce"
            )
            # E_Flux = dtmp / (constants.GRAV * pdt * 4. * constants.PI *
            # constants.RADIUS**2)
        elif consv < -constants.CONSV_MIN:
            sum_z1(
                pkz,
                delp,
                te0_2d,
                te_2d,
                zsum1,
                origin=grid.compute_origin(),
                domain=grid.domain_shape_compute(),
            )
            # E_Flux = consv
            # dtmp  = E_Flux *  (constants.GRAV * pdt * 4. * constants.PI *
            # constants.RADIUS**2) / (constants.CV_AIR * g_sum(zsum1,
            # grid,.area_64))
            raise Exception(
                "Unimplemented/untested case consv("
                + str(consv)
                + ")  < -CONSV_MIN("
                + str(-constants.CONSV_MIN)
                + ")"
            )

    if spec.namelist.do_sat_adj:

        kmp_origin = (grid.is_, grid.js, kmp)
        kmp_domain = (grid.nic, grid.njc, grid.npz - kmp)
        saturation_adjustment.compute(
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
            kmp,
        )
        if fast_mp_consv:
            sum_te(te, te0_2d, origin=kmp_origin, domain=kmp_domain)
    if last_step:
        moist_cv.compute_last_step(
            pt, pkz, dtmp, r_vir, qvapor, qliquid, qice, qrain, qsnow, qgraupel, gz
        )
    else:
        basic.adjust_divide_stencil(
            pkz, pt, origin=grid.compute_origin(), domain=grid.domain_shape_compute()
        )
