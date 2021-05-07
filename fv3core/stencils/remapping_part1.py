from typing import Any, Dict

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

import fv3core._config as spec
import fv3core.stencils.moist_cv as moist_cv
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil, gtstencil
from fv3core.stencils.map_single import MapSingle
from fv3core.stencils.mapn_tracer import MapNTracer
from fv3core.stencils.moist_cv import moist_pt_func
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


def update_ua(pe2: FloatField, ua: FloatField):
    with computation(PARALLEL), interval(0, -1):
        ua = pe2[0, 0, 1]


class VerticalRemapping1:
    def __init__(self, namelist, nq):
        """
        Test is Remapping_Part1
        Fortran code is the first section of lagrangian_to_eulerian in fv_mapz.F90
        """

        if namelist.kord_tm >= 0:
            raise Exception("map ppm, untested mode where kord_tm >= 0")

        hydrostatic = namelist.hydrostatic
        if hydrostatic:
            raise Exception("Hydrostatic is not implemented")

        grid = spec.grid
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

        self._init_pe = FrozenStencil(
            init_pe, origin=grid.compute_origin(), domain=self._domain_jextra
        )

        self._moist_cv_pt_pressure = FrozenStencil(
            moist_cv_pt_pressure,
            externals={"kord_tm": namelist.kord_tm, "hydrostatic": hydrostatic},
            origin=grid.compute_origin(),
            domain=grid.domain_shape_compute(add=(0, 0, 1)),
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

        kord_tm = abs(namelist.kord_tm)
        self._map_single_pt = MapSingle(kord_tm, 1, grid.is_, grid.ie, grid.js, grid.je)

        self._mapn_tracer = MapNTracer(
            abs(namelist.kord_tr), nq, grid.is_, grid.ie, grid.js, grid.je
        )

        kord_wz = namelist.kord_wz
        self._map_single_w = MapSingle(kord_wz, -2, grid.is_, grid.ie, grid.js, grid.je)
        self._map_single_delz = MapSingle(
            kord_wz, 1, grid.is_, grid.ie, grid.js, grid.je
        )

        self._undo_delz_adjust_and_copy_peln = FrozenStencil(
            undo_delz_adjust_and_copy_peln,
            origin=grid.compute_origin(),
            domain=(grid.nic, grid.njc, grid.npz + 1),
        )

        self._pressures_mapu = FrozenStencil(
            pressures_mapu, origin=grid.compute_origin(), domain=self._domain_jextra
        )

        kord_mt = namelist.kord_mt
        self._map_single_u = MapSingle(
            kord_mt, -1, grid.is_, grid.ie, grid.js, grid.je + 1
        )

        domain_iextra = (grid.nic + 1, grid.njc, grid.npz + 1)
        self._pressures_mapv = FrozenStencil(
            pressures_mapv, origin=grid.compute_origin(), domain=domain_iextra
        )

        self._map_single_v = MapSingle(
            kord_mt, -1, grid.is_, grid.ie + 1, grid.js, grid.je
        )

        self._update_ua = FrozenStencil(
            update_ua, origin=grid.compute_origin(), domain=self._domain_jextra
        )

    def __call__(
        self,
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
    ):
        # TODO: Many of these could be passed at runtime
        """
        Remaps tracers, winds, pt, and delz from the deformed Lagrangian grid
        to the Eulerian grid.
        """

        akap = constants.KAPPA
        r_vir = constants.ZVIR

        self._init_pe(pe, self._pe1, self._pe2, ptop)

        self._moist_cv_pt_pressure(
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
            self._pe2,
            ak,
            bk,
            self._dp2,
            ps,
            self._pn2,
            peln,
            r_vir,
        )

        # TODO: Fix silly hack due to pe2 being 2d, so pe[:, je+1, 1:npz] should be
        # the same as it was for pe[:, je, 1:npz] (unchanged)
        self._copy_j_adjacent(self._pe2)

        self._pn2_pk_delp(self._dp2, delp, self._pe2, self._pn2, pk, akap)

        self._map_single_pt(pt, peln, self._pn2, gz, qmin=self._t_min)

        # TODO if self._nq > 5:
        self._mapn_tracer(self._pe1, self._pe2, self._dp2, tracers, 0.0)
        # TODO else if self._nq > 0:
        # TODO map1_q2, fillz

        self._map_single_w(w, self._pe1, self._pe2, wsd)
        self._map_single_delz(delz, self._pe1, self._pe2, gz)

        self._undo_delz_adjust_and_copy_peln(delp, delz, peln, self._pe0, self._pn2)
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

        self._pressures_mapu(pe, self._pe1, ak, bk, self._pe0, self._pe3)
        self._map_single_u(u, self._pe0, self._pe3, gz)

        self._pressures_mapv(pe, ak, bk, self._pe0, self._pe3)
        self._map_single_v(v, self._pe0, self._pe3, gz)

        self._update_ua(self._pe2, ua)
