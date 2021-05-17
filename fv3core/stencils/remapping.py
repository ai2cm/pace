from typing import Dict

from gt4py.gtscript import (
    __INLINED,
    BACKWARD,
    FORWARD,
    PARALLEL,
    computation,
    exp,
    interval,
    log,
)

import fv3core.stencils.moist_cv as moist_cv
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil
from fv3core.stencils.basic_operations import adjust_divide_stencil
from fv3core.stencils.map_single import MapSingle
from fv3core.stencils.mapn_tracer import MapNTracer
from fv3core.stencils.moist_cv import moist_pt_func, moist_pt_last_step
from fv3core.stencils.saturation_adjustment import SatAdjust3d
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK


CONSV_MIN = 0.001


def init_pe(pe: FloatField, pe1: FloatField, pe2: FloatField, ptop: float):
    with computation(PARALLEL):
        with interval(0, 1):
            pe2 = ptop
        with interval(-1, None):
            pe2 = pe
    with computation(PARALLEL), interval(...):
        pe1 = pe


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
):
    from __externals__ import hydrostatic, kord_tm

    # moist_cv.moist_pt
    with computation(PARALLEL), interval(0, -1):
        if __INLINED(kord_tm < 0):
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
        if __INLINED(not hydrostatic):
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


def copy_j_adjacent(pe2: FloatField):
    with computation(PARALLEL), interval(...):
        pe2_0 = pe2[0, -1, 0]
        pe2 = pe2_0


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


def update_ua(pe2: FloatField, ua: FloatField):
    with computation(PARALLEL), interval(0, -1):
        ua = pe2[0, 0, 1]


def copy_from_below(a: FloatField, b: FloatField):
    with computation(PARALLEL), interval(1, None):
        b = a[0, 0, -1]


def sum_te(te: FloatField, te0_2d: FloatField):
    with computation(FORWARD):
        with interval(0, None):
            te0_2d = te0_2d[0, 0, -1] + te


class Lagrangian_to_Eulerian:
    def __init__(self, grid, namelist, nq, pfull):
        if namelist.kord_tm >= 0:
            raise Exception("map ppm, untested mode where kord_tm >= 0")

        hydrostatic = namelist.hydrostatic
        if hydrostatic:
            raise Exception("Hydrostatic is not implemented")

        shape_kplus = grid.domain_shape_full(add=(0, 0, 1))
        self._t_min = 184.0
        self._nq = nq
        # do_omega = hydrostatic and last_step # TODO pull into inputs
        self._domain_jextra = (grid.nic, grid.njc + 1, grid.npz + 1)

        self._pe1 = utils.make_storage_from_shape(shape_kplus)
        self._pe2 = utils.make_storage_from_shape(shape_kplus)
        self._dp2 = utils.make_storage_from_shape(shape_kplus)
        self._pn2 = utils.make_storage_from_shape(shape_kplus)
        self._pe0 = utils.make_storage_from_shape(shape_kplus)
        self._pe3 = utils.make_storage_from_shape(shape_kplus)

        self._gz: FloatField = utils.make_storage_from_shape(
            shape_kplus, grid.compute_origin()
        )
        self._cvm: FloatField = utils.make_storage_from_shape(
            shape_kplus, grid.compute_origin()
        )

        self._init_pe = FrozenStencil(
            init_pe, origin=grid.compute_origin(), domain=self._domain_jextra
        )

        self._moist_cv_pt_pressure = FrozenStencil(
            moist_cv_pt_pressure,
            externals={"kord_tm": namelist.kord_tm, "hydrostatic": hydrostatic},
            origin=grid.compute_origin(),
            domain=grid.domain_shape_compute(add=(0, 0, 1)),
        )
        self._moist_cv_pkz = FrozenStencil(
            moist_cv.moist_pkz,
            origin=grid.compute_origin(),
            domain=grid.domain_shape_compute(),
        )
        self._copy_j_adjacent = FrozenStencil(
            copy_j_adjacent,
            origin=(grid.is_, grid.je + 1, 1),
            domain=(grid.nic, 1, grid.npz - 1),
        )

        self._pn2_pk_delp = FrozenStencil(
            pn2_pk_delp,
            origin=grid.compute_origin(),
            domain=grid.domain_shape_compute(),
        )

        self._kord_tm = abs(namelist.kord_tm)
        self._map_single_pt = MapSingle(
            self._kord_tm, 1, grid.is_, grid.ie, grid.js, grid.je
        )

        self._mapn_tracer = MapNTracer(
            abs(namelist.kord_tr), nq, grid.is_, grid.ie, grid.js, grid.je
        )

        self._kord_wz = namelist.kord_wz
        self._map_single_w = MapSingle(
            self._kord_wz, -2, grid.is_, grid.ie, grid.js, grid.je
        )
        self._map_single_delz = MapSingle(
            self._kord_wz, 1, grid.is_, grid.ie, grid.js, grid.je
        )

        self._undo_delz_adjust_and_copy_peln = FrozenStencil(
            undo_delz_adjust_and_copy_peln,
            origin=grid.compute_origin(),
            domain=(grid.nic, grid.njc, grid.npz + 1),
        )

        self._pressures_mapu = FrozenStencil(
            pressures_mapu, origin=grid.compute_origin(), domain=self._domain_jextra
        )

        self._kord_mt = namelist.kord_mt
        self._map_single_u = MapSingle(
            self._kord_mt, -1, grid.is_, grid.ie, grid.js, grid.je + 1
        )

        self._pressures_mapv = FrozenStencil(
            pressures_mapv,
            origin=grid.compute_origin(),
            domain=(grid.nic + 1, grid.njc, grid.npz + 1),
        )

        self._map_single_v = MapSingle(
            self._kord_mt, -1, grid.is_, grid.ie + 1, grid.js, grid.je
        )

        self._update_ua = FrozenStencil(
            update_ua, origin=grid.compute_origin(), domain=self._domain_jextra
        )

        self._copy_from_below_stencil = FrozenStencil(
            copy_from_below,
            origin=grid.compute_origin(),
            domain=grid.domain_shape_compute(),
        )

        self._moist_cv_last_step_stencil = FrozenStencil(
            moist_pt_last_step,
            origin=(grid.is_, grid.js, 0),
            domain=(grid.nic, grid.njc, grid.npz + 1),
        )

        self._basic_adjust_divide_stencil = FrozenStencil(
            adjust_divide_stencil,
            origin=grid.compute_origin(),
            domain=grid.domain_shape_compute(),
        )

        self._do_sat_adjust = namelist.do_sat_adj

        self.kmp = grid.npz - 1
        for k in range(pfull.shape[0]):
            if pfull[k] > 10.0e2:
                self.kmp = k
                break

        self._saturation_adjustment = SatAdjust3d(self.kmp)

        self._sum_te_stencil = FrozenStencil(
            sum_te,
            origin=(grid.is_, grid.js, self.kmp),
            domain=(grid.nic, grid.njc, grid.npz - self.kmp),
        )

    def __call__(
        self,
        tracers: Dict[str, "FloatField"],
        pt: FloatField,
        delp: FloatField,
        delz: FloatField,
        peln: FloatField,
        u: FloatField,
        v: FloatField,
        w: FloatField,
        ua: FloatField,
        va: FloatField,
        cappa: FloatField,
        q_con: FloatField,
        q_cld: FloatField,
        pkz: FloatField,
        pk: FloatField,
        pe: FloatField,
        hs: FloatFieldIJ,
        te0_2d: FloatFieldIJ,
        ps: FloatFieldIJ,
        wsd: FloatFieldIJ,
        omga: FloatField,
        ak: FloatFieldK,
        bk: FloatFieldK,
        pfull: FloatFieldK,
        dp1: FloatField,
        ptop: float,
        akap: float,
        zvir: float,
        last_step: bool,
        consv_te: float,
        mdt: float,
        bdt: float,
        do_adiabatic_init: bool,
        nq: int,
    ):
        """
        pt: D-grid potential temperature (inout)
        delp: Pressure Thickness (inout)
        delz: Vertical thickness of atmosphere layers (in)
        peln: Logarithm of interface pressure (inout)
        u: D-grid x-velocity (inout)
        v: D-grid y-velocity (inout)
        w: vertical velocity (inout)
        ua: A-grid x-velocity (inout)
        va: A-grid y-velocity (inout)
        cappa: Power to raise pressure to (inout)
        q_con: total condensate mixing ratio (inout)
        q_cld:
        pkz: Layer mean pressure raised to the power of Kappa (in)
        pk: interface pressure raised to power of kappa, final acoustic value (inout)
        pe: pressure at layer edges (inout)
        hs: surface geopotential (in)
        te0_2d:
        ps: surface pressure (inout)
        wsd:
        omga: Vertical pressure velocity (inout)
        ak: (in)
        bk (in)
        pfull: (in)
        dp1:
        ptop: (in)
        akap: (in)
        zvir:
        last_step
        consv_te
        mdt : remap time step (in)
        bdt
        do_adiabatic_init
        nq: number of tracers (in)

        Remap the deformed Lagrangian surfaces onto the reference, or "Eulerian",
        coordinate levels.
        """
        self._init_pe(pe, self._pe1, self._pe2, ptop)

        self._moist_cv_pt_pressure(
            tracers["qvapor"],
            tracers["qliquid"],
            tracers["qrain"],
            tracers["qsnow"],
            tracers["qice"],
            tracers["qgraupel"],
            q_con,
            self._gz,
            self._cvm,
            pt,
            cappa,
            delp,
            delz,
            pe,
            self._pe2,
            ak,
            bk,
            self._dp2,
            ps,
            self._pn2,
            peln,
            zvir,
        )

        # TODO: Fix silly hack due to pe2 being 2d, so pe[:, je+1, 1:npz] should be
        # the same as it was for pe[:, je, 1:npz] (unchanged)
        self._copy_j_adjacent(self._pe2)

        self._pn2_pk_delp(self._dp2, delp, self._pe2, self._pn2, pk, akap)

        self._map_single_pt(pt, peln, self._pn2, qmin=self._t_min)

        # TODO if self._nq > 5:
        self._mapn_tracer(self._pe1, self._pe2, self._dp2, tracers, 0.0)
        # TODO else if self._nq > 0:
        # TODO map1_q2, fillz

        self._map_single_w(w, self._pe1, self._pe2, qs=wsd)
        self._map_single_delz(delz, self._pe1, self._pe2)

        self._undo_delz_adjust_and_copy_peln(delp, delz, peln, self._pe0, self._pn2)
        # if do_omega:  # NOTE untested
        #    pe3 = copy(omga, origin=(grid.is_, grid.js, 1))

        self._moist_cv_pkz(
            tracers["qvapor"],
            tracers["qliquid"],
            tracers["qrain"],
            tracers["qsnow"],
            tracers["qice"],
            tracers["qgraupel"],
            q_con,
            self._gz,
            self._cvm,
            pkz,
            pt,
            cappa,
            delp,
            delz,
            zvir,
        )

        # if do_omega:
        # dp2 update, if larger than pe0 and smaller than one level up, update omega
        # and exit

        self._pressures_mapu(pe, self._pe1, ak, bk, self._pe0, self._pe3)
        self._map_single_u(u, self._pe0, self._pe3)

        self._pressures_mapv(pe, ak, bk, self._pe0, self._pe3)
        self._map_single_v(v, self._pe0, self._pe3)

        self._update_ua(self._pe2, ua)

        self._copy_from_below_stencil(ua, pe)
        dtmp = 0.0
        if last_step and not do_adiabatic_init:
            if consv_te > constants.CONSV_MIN:
                raise NotImplementedError(
                    "We do not support consv_te > 0.001 "
                    "because that would trigger an allReduce"
                )
            elif consv_te < -constants.CONSV_MIN:
                raise Exception(
                    "Unimplemented/untested case consv("
                    + str(consv_te)
                    + ")  < -CONSV_MIN("
                    + str(-constants.CONSV_MIN)
                    + ")"
                )

        if self._do_sat_adjust:
            fast_mp_consv = not do_adiabatic_init and consv_te > constants.CONSV_MIN
            self._saturation_adjustment(
                dp1,
                tracers["qvapor"],
                tracers["qliquid"],
                tracers["qice"],
                tracers["qrain"],
                tracers["qsnow"],
                tracers["qgraupel"],
                q_cld,
                hs,
                peln,
                delp,
                delz,
                q_con,
                pt,
                pkz,
                cappa,
                zvir,
                mdt,
                fast_mp_consv,
                last_step,
                akap,
                self.kmp,
            )
            if fast_mp_consv:
                self._sum_te_stencil(
                    dp1,
                    te0_2d,
                )

        if last_step:
            self._moist_cv_last_step_stencil(
                tracers["qvapor"],
                tracers["qliquid"],
                tracers["qrain"],
                tracers["qsnow"],
                tracers["qice"],
                tracers["qgraupel"],
                self._gz,
                pt,
                pkz,
                dtmp,
                zvir,
            )
        else:
            self._basic_adjust_divide_stencil(pkz, pt)
