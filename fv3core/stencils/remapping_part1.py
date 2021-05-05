from typing import Any, Dict

from gt4py.gtscript import BACKWARD, FORWARD, PARALLEL, computation, exp, interval, log

import fv3core._config as spec
import fv3core.stencils.mapn_tracer as mapn_tracer
import fv3core.stencils.moist_cv as moist_cv
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.map_single import MapSingle
from fv3core.stencils.moist_cv import moist_pt_func
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK


CONSV_MIN = 0.001


@gtstencil
def init_pe(pe: FloatField, pe1: FloatField, pe2: FloatField, ptop: float):
    with computation(PARALLEL):
        with interval(0, 1):
            pe2 = ptop
        with interval(-1, None):
            pe2 = pe
    with computation(PARALLEL), interval(...):
        pe1 = pe


@gtstencil
def undo_delz_adjust_and_copy_peln(
    delp: FloatField,
    delz: FloatField,
    peln: FloatField,
    pe0: FloatField,
    pn2: FloatField,
):
    with computation(PARALLEL), interval(0, -1):
        delz = -delz * delp
    with computation(PARALLEL), interval(...):
        pe0 = peln
        peln = pn2


@gtstencil
def moist_cv_pt_pressure(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qice: FloatField,
    qgraupel: FloatField,
    q_con: FloatField,
    gz: FloatField,
    cvm: FloatField,
    pt: FloatField,
    cappa: FloatField,
    delp: FloatField,
    delz: FloatField,
    pe: FloatField,
    pe2: FloatField,
    ak: FloatFieldK,
    bk: FloatFieldK,
    dp2: FloatField,
    ps: FloatFieldIJ,
    pn2: FloatField,
    peln: FloatField,
    r_vir: float,
    hydrostatic: bool,
):
    from __externals__ import namelist

    # moist_cv.moist_pt
    with computation(PARALLEL), interval(0, -1):
        if namelist.kord_tm < 0:
            cvm, gz, q_con, cappa, pt = moist_pt_func(
                qvapor,
                qliquid,
                qrain,
                qsnow,
                qice,
                qgraupel,
                q_con,
                gz,
                cvm,
                pt,
                cappa,
                delp,
                delz,
                r_vir,
            )
        # delz_adjust
        if not hydrostatic:
            delz = -delz / delp
    # pressure_updates
    with computation(FORWARD):
        with interval(-1, None):
            ps = pe
    with computation(PARALLEL):
        with interval(0, 1):
            pn2 = peln
        with interval(1, -1):
            pe2 = ak + bk * ps
        with interval(-1, None):
            pn2 = peln
    with computation(BACKWARD), interval(0, -1):
        dp2 = pe2[0, 0, 1] - pe2
    # copy_stencil
    with computation(PARALLEL), interval(0, -1):
        delp = dp2


@gtstencil
def copy_j_adjacent(pe2: FloatField):
    with computation(PARALLEL), interval(...):
        pe2_0 = pe2[0, -1, 0]
        pe2 = pe2_0


@gtstencil
def pn2_pk_delp(
    dp2: FloatField,
    delp: FloatField,
    pe2: FloatField,
    pn2: FloatField,
    pk: FloatField,
    akap: float,
):
    with computation(PARALLEL), interval(...):
        delp = dp2
        pn2 = log(pe2)
        pk = exp(akap * pn2)


@gtstencil
def pressures_mapu(
    pe: FloatField,
    pe1: FloatField,
    ak: FloatFieldK,
    bk: FloatFieldK,
    pe0: FloatField,
    pe3: FloatField,
):
    with computation(BACKWARD):
        with interval(-1, None):
            pe_bottom = pe
            pe1_bottom = pe
        with interval(0, -1):
            pe_bottom = pe_bottom[0, 0, 1]
            pe1_bottom = pe1_bottom[0, 0, 1]
    with computation(FORWARD):
        with interval(0, 1):
            pe0 = pe
        with interval(1, None):
            pe0 = 0.5 * (pe[0, -1, 0] + pe1)
    with computation(FORWARD), interval(...):
        bkh = 0.5 * bk
        pe3 = ak + bkh * (pe_bottom[0, -1, 0] + pe1_bottom)


@gtstencil
def pressures_mapv(
    pe: FloatField, ak: FloatFieldK, bk: FloatFieldK, pe0: FloatField, pe3: FloatField
):
    with computation(BACKWARD):
        with interval(-1, None):
            pe_bottom = pe
        with interval(0, -1):
            pe_bottom = pe_bottom[0, 0, 1]
    with computation(FORWARD):
        with interval(0, 1):
            pe3 = ak
            pe0 = pe
        with interval(1, None):
            bkh = 0.5 * bk
            pe0 = 0.5 * (pe[-1, 0, 0] + pe)
            pe3 = ak + bkh * (pe_bottom[-1, 0, 0] + pe_bottom)


@gtstencil
def update_ua(pe2: FloatField, ua: FloatField):
    with computation(PARALLEL), interval(0, -1):
        ua = pe2[0, 0, 1]


def compute(
    tracers: Dict[str, Any],
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
    pkz: FloatField,
    pk: FloatField,
    pe: FloatField,
    hs: FloatFieldIJ,
    te: FloatField,
    ps: FloatFieldIJ,
    wsd: FloatField,
    omga: FloatField,
    ak: FloatFieldK,
    bk: FloatFieldK,
    gz: FloatField,
    cvm: FloatField,
    ptop: float,
    akap: float,
    r_vir: float,
    nq: int,
):
    if spec.namelist.kord_tm >= 0:
        raise Exception("map ppm, untested mode where kord_tm >= 0")

    hydrostatic = spec.namelist.hydrostatic
    if hydrostatic:
        raise Exception("Hydrostatic is not implemented")

    grid = spec.grid
    t_min = 184.0

    # do_omega = hydrostatic and last_step # TODO pull into inputs
    domain_jextra = (grid.nic, grid.njc + 1, grid.npz + 1)

    pe1 = utils.make_storage_from_shape(
        pe.shape, grid.compute_origin(), cache_key="remapping_part1_pe1"
    )

    pe2 = utils.make_storage_from_shape(
        pe.shape, grid.compute_origin(), cache_key="remapping_part1_pe2"
    )
    dp2 = utils.make_storage_from_shape(
        pe.shape, grid.compute_origin(), cache_key="remapping_part1_dp2"
    )
    pn2 = utils.make_storage_from_shape(
        pe.shape, grid.compute_origin(), cache_key="remapping_part1_pn2"
    )
    pe0 = utils.make_storage_from_shape(
        pe.shape, grid.compute_origin(), cache_key="remapping_part1_pe0"
    )
    pe3 = utils.make_storage_from_shape(
        pe.shape, grid.compute_origin(), cache_key="remapping_part1_pe3"
    )

    init_pe(pe, pe1, pe2, ptop, origin=grid.compute_origin(), domain=domain_jextra)

    moist_cv_pt_pressure(
        tracers["qvapor"],
        tracers["qliquid"],
        tracers["qrain"],
        tracers["qsnow"],
        tracers["qice"],
        tracers["qgraupel"],
        q_con,
        gz,
        cvm,
        pt,
        cappa,
        delp,
        delz,
        pe,
        pe2,
        ak,
        bk,
        dp2,
        ps,
        pn2,
        peln,
        r_vir,
        hydrostatic,
        origin=grid.compute_origin(),
        domain=grid.domain_shape_compute(add=(0, 0, 1)),
    )

    # TODO: Fix silly hack due to pe2 being 2d, so pe[:, je+1, 1:npz] should be
    # the same as it was for pe[:, je, 1:npz] (unchanged)
    copy_j_adjacent(
        pe2, origin=(grid.is_, grid.je + 1, 1), domain=(grid.nic, 1, grid.npz - 1)
    )
    pn2_pk_delp(
        dp2,
        delp,
        pe2,
        pn2,
        pk,
        akap,
        origin=grid.compute_origin(),
        domain=grid.domain_shape_compute(),
    )

    kord_tm = abs(spec.namelist.kord_tm)
    map_single = utils.cached_stencil_class(MapSingle)(
        kord_tm, 1, grid.is_, grid.ie, grid.js, grid.je, cache_key="remap1-single1"
    )
    map_single(
        pt,
        peln,
        pn2,
        gz,
        qmin=t_min,
    )

    # TODO if nq > 5:
    mapn_tracer.compute(
        pe1,
        pe2,
        dp2,
        tracers,
        nq,
        0.0,
        grid.is_,
        grid.ie,
        grid.js,
        grid.je,
        abs(spec.namelist.kord_tr),
    )
    # TODO else if nq > 0:
    # TODO map1_q2, fillz
    kord_wz = spec.namelist.kord_wz
    map_single = utils.cached_stencil_class(MapSingle)(
        kord_wz, -2, grid.is_, grid.ie, grid.js, grid.je, cache_key="remap1-single2"
    )
    map_single(w, pe1, pe2, wsd)
    map_single = utils.cached_stencil_class(MapSingle)(
        kord_wz, 1, grid.is_, grid.ie, grid.js, grid.je, cache_key="remap1-single3"
    )
    map_single(delz, pe1, pe2, gz)

    undo_delz_adjust_and_copy_peln(
        delp,
        delz,
        peln,
        pe0,
        pn2,
        origin=grid.compute_origin(),
        domain=(grid.nic, grid.njc, grid.npz + 1),
    )
    # if do_omega:  # NOTE untested
    #    pe3 = copy(omga, origin=(grid.is_, grid.js, 1))

    moist_cv.compute_pkz(
        tracers["qvapor"],
        tracers["qliquid"],
        tracers["qice"],
        tracers["qrain"],
        tracers["qsnow"],
        tracers["qgraupel"],
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
    # if do_omega:
    # dp2 update, if larger than pe0 and smaller than one level up, update omega
    # and exit

    pressures_mapu(
        pe, pe1, ak, bk, pe0, pe3, origin=grid.compute_origin(), domain=domain_jextra
    )
    kord_mt = spec.namelist.kord_mt
    map_single = utils.cached_stencil_class(MapSingle)(
        kord_mt, -1, grid.is_, grid.ie, grid.js, grid.je + 1, cache_key="remap1-single4"
    )
    map_single(u, pe0, pe3, gz)
    domain_iextra = (grid.nic + 1, grid.njc, grid.npz + 1)
    pressures_mapv(
        pe, ak, bk, pe0, pe3, origin=grid.compute_origin(), domain=domain_iextra
    )
    map_single = utils.cached_stencil_class(MapSingle)(
        kord_mt, -1, grid.is_, grid.ie + 1, grid.js, grid.je, cache_key="remap1-single5"
    )
    map_single(v, pe0, pe3, gz)
    update_ua(pe2, ua, origin=grid.compute_origin(), domain=domain_jextra)
