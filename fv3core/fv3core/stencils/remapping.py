from typing import Dict

from gt4py.gtscript import (
    __INLINED,
    BACKWARD,
    FORWARD,
    PARALLEL,
    computation,
    exp,
    horizontal,
    interval,
    log,
    region,
)

import fv3core.stencils.moist_cv as moist_cv
import pace.dsl.gt4py_utils as utils
from fv3core._config import RemappingConfig
from fv3core.stencils.basic_operations import adjust_divide_stencil
from fv3core.stencils.map_single import MapSingle
from fv3core.stencils.mapn_tracer import MapNTracer
from fv3core.stencils.moist_cv import moist_pt_func, moist_pt_last_step
from fv3core.stencils.saturation_adjustment import SatAdjust3d
from pace.dsl.dace.orchestration import orchestrate
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ, FloatFieldK
from pace.util import Quantity


# TODO: Should this be set here or in global_constants?
CONSV_MIN = 0.001


def init_pe(pe: FloatField, pe1: FloatField, pe2: FloatField, ptop: float):
    """
    Args:
        pe (in):
        pe1 (out):
        pe2 (out):
        ptop (in):
    """
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
    """
    Args:
        delp (in):
        delz (inout):
        peln (inout):
        pe0 (out):
        pn2 (in):
    """
    # TODO: We can assign pe0 and peln outside of a stencil to save the data copying
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
    """
    Args:
        qvapor (in):
        qliquid (in):
        qrain (in):
        qsnow (in):
        qice (in):
        qgraupel (in):
        q_con (out):
        gz (out):
        cvm (out):
        pt (inout):
        cappa (out):
        delp (inout):
        delz (inout):
        pe (in):
        pe2 (inout):
        ak (in):
        bk (in):
        dp2 (out):
        ps (out):
        pn2 (out):
        peln (in):
    """
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


def pn2_pk_delp(
    dp2: FloatField,
    delp: FloatField,
    pe2: FloatField,
    pn2: FloatField,
    pk: FloatField,
    akap: float,
):
    """
    Args:
        dp2 (in):
        delp (out):
        pe2 (in):
        pn2 (out):
        pk (out):
    """
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
    """
    Args:
        pe (in):
        pe1 (in):
        ak (in):
        bk (in):
        pe0 (out):
        pe3 (out):
    """
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
    """
    Args:
        pe (in):
        ak (in):
        bk (in):
        pe0 (out):
        pe3 (out):
    """
    # TODO: Combine pressures_mapu and pressures_mapv
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
    """
    Args:
        pe2 (in):
        ua (out):
    """
    from __externals__ import local_je

    with computation(PARALLEL), interval(0, -1):
        ua = pe2[0, 0, 1]

    # pe2[:, je+1, 1:npz] should equal pe2[:, je, 1:npz] as in the Fortran model,
    # but the extra j-elements are only used here, so we can just directly assign ua.
    # Maybe we can eliminate this later?
    with computation(PARALLEL), interval(0, -2):
        with horizontal(region[:, local_je + 1]):
            ua = pe2[0, -1, 1]


def copy_from_below(a: FloatField, b: FloatField):
    """
    Args:
        a (in):
        b (out):
    """
    with computation(PARALLEL), interval(1, None):
        b = a[0, 0, -1]


class LagrangianToEulerian:
    """
    Fortran name is Lagrangian_to_Eulerian
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        config: RemappingConfig,
        area_64,
        nq,
        pfull,
        tracers: Dict[str, Quantity],
    ):
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            dace_constant_args=["tracers"],
        )
        grid_indexing = stencil_factory.grid_indexing
        if config.kord_tm >= 0:
            raise NotImplementedError("map ppm, untested mode where kord_tm >= 0")
        hydrostatic = config.hydrostatic
        if hydrostatic:
            raise NotImplementedError("Hydrostatic is not implemented")

        shape_kplus = grid_indexing.domain_full(add=(0, 0, 1))
        self._t_min = 184.0
        self._nq = nq
        # do_omega = hydrostatic and last_step # TODO pull into inputs
        self._domain_jextra = (
            grid_indexing.domain[0],
            grid_indexing.domain[1] + 1,
            grid_indexing.domain[2] + 1,
        )

        backend = stencil_factory.backend
        self._pe1 = utils.make_storage_from_shape(shape_kplus, backend=backend)
        self._pe2 = utils.make_storage_from_shape(shape_kplus, backend=backend)
        self._dp2 = utils.make_storage_from_shape(shape_kplus, backend=backend)
        self._pn2 = utils.make_storage_from_shape(shape_kplus, backend=backend)
        self._pe0 = utils.make_storage_from_shape(shape_kplus, backend=backend)
        self._pe3 = utils.make_storage_from_shape(shape_kplus, backend=backend)

        self._gz: FloatField = utils.make_storage_from_shape(
            shape_kplus, grid_indexing.origin_compute(), backend=backend
        )
        self._cvm: FloatField = utils.make_storage_from_shape(
            shape_kplus, grid_indexing.origin_compute(), backend=backend
        )

        self._kord_tm = abs(config.kord_tm)
        self._kord_wz = config.kord_wz
        self._kord_mt = config.kord_mt

        self._do_sat_adjust = config.do_sat_adj

        self.kmp = grid_indexing.domain[2] - 1
        for k in range(pfull.shape[0]):
            if pfull[k] > 10.0e2:
                self.kmp = k
                break

        self._init_pe = stencil_factory.from_origin_domain(
            init_pe, origin=grid_indexing.origin_compute(), domain=self._domain_jextra
        )

        self._moist_cv_pt_pressure = stencil_factory.from_origin_domain(
            moist_cv_pt_pressure,
            externals={"kord_tm": config.kord_tm, "hydrostatic": hydrostatic},
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(add=(0, 0, 1)),
        )

        self._pn2_pk_delp = stencil_factory.from_origin_domain(
            pn2_pk_delp,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._map_single_pt = MapSingle(
            stencil_factory,
            self._kord_tm,
            1,
            grid_indexing.isc,
            grid_indexing.iec,
            grid_indexing.jsc,
            grid_indexing.jec,
        )

        self._mapn_tracer = MapNTracer(
            stencil_factory,
            abs(config.kord_tr),
            nq,
            grid_indexing.isc,
            grid_indexing.iec,
            grid_indexing.jsc,
            grid_indexing.jec,
            fill=config.fill,
            tracers=tracers,
        )

        self._map_single_w = MapSingle(
            stencil_factory,
            self._kord_wz,
            -2,
            grid_indexing.isc,
            grid_indexing.iec,
            grid_indexing.jsc,
            grid_indexing.jec,
        )

        self._map_single_delz = MapSingle(
            stencil_factory,
            self._kord_wz,
            1,
            grid_indexing.isc,
            grid_indexing.iec,
            grid_indexing.jsc,
            grid_indexing.jec,
        )

        self._undo_delz_adjust_and_copy_peln = stencil_factory.from_origin_domain(
            undo_delz_adjust_and_copy_peln,
            origin=grid_indexing.origin_compute(),
            domain=(
                grid_indexing.domain[0],
                grid_indexing.domain[1],
                grid_indexing.domain[2] + 1,
            ),
        )

        self._moist_cv_pkz = stencil_factory.from_origin_domain(
            moist_cv.moist_pkz,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._pressures_mapu = stencil_factory.from_origin_domain(
            pressures_mapu,
            origin=grid_indexing.origin_compute(),
            domain=self._domain_jextra,
        )

        self._map_single_u = MapSingle(
            stencil_factory,
            self._kord_mt,
            -1,
            grid_indexing.isc,
            grid_indexing.iec,
            grid_indexing.jsc,
            grid_indexing.jec + 1,
        )

        self._pressures_mapv = stencil_factory.from_origin_domain(
            pressures_mapv,
            origin=grid_indexing.origin_compute(),
            domain=(
                grid_indexing.domain[0] + 1,
                grid_indexing.domain[1],
                grid_indexing.domain[2] + 1,
            ),
        )

        self._map_single_v = MapSingle(
            stencil_factory,
            self._kord_mt,
            -1,
            grid_indexing.isc,
            grid_indexing.iec + 1,
            grid_indexing.jsc,
            grid_indexing.jec,
        )

        ax_offsets_jextra = grid_indexing.axis_offsets(
            grid_indexing.origin_compute(),
            self._domain_jextra,
        )
        self._update_ua = stencil_factory.from_origin_domain(
            update_ua,
            origin=grid_indexing.origin_compute(),
            domain=self._domain_jextra,
            externals={**ax_offsets_jextra},
        )

        self._copy_from_below_stencil = stencil_factory.from_origin_domain(
            copy_from_below,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._saturation_adjustment = SatAdjust3d(
            stencil_factory, config.sat_adjust, area_64, self.kmp
        )

        self._moist_cv_last_step_stencil = stencil_factory.from_origin_domain(
            moist_pt_last_step,
            origin=(grid_indexing.isc, grid_indexing.jsc, 0),
            domain=(
                grid_indexing.domain[0],
                grid_indexing.domain[1],
                grid_indexing.domain[2] + 1,
            ),
        )

        self._basic_adjust_divide_stencil = stencil_factory.from_origin_domain(
            adjust_divide_stencil,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

    def __call__(
        self,
        tracers: Dict[str, Quantity],
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
    ):
        """
        tracers (inout): Tracer species tracked across
        pt (inout): D-grid potential temperature
        delp (inout): Pressure Thickness
        delz (in): Vertical thickness of atmosphere layers
        peln (inout): Logarithm of interface pressure
        u (inout): D-grid x-velocity
        v (inout): D-grid y-velocity
        w (inout): Vertical velocity
        ua (inout): A-grid x-velocity
        va (inout): A-grid y-velocity
        cappa (inout): Power to raise pressure to
        q_con (out): Total condensate mixing ratio
        q_cld (out): Cloud fraction
        pkz (in): Layer mean pressure raised to the power of Kappa
        pk (out): Interface pressure raised to power of kappa, final acoustic value
        pe (in): Pressure at layer edges
        hs (in): Surface geopotential
        te0_2d (unused): Atmosphere total energy in columns
        ps (out): Surface pressure
        wsd (in): Vertical velocity of the lowest level
        omga (unused): Vertical pressure velocity
        ak (in): Atmosphere hybrid a coordinate (Pa)
        bk (in): Atmosphere hybrid b coordinate (dimensionless)
        pfull (in): Pressure full levels
        dp1 (out): Pressure thickness before dyn_core (only written
            if do_sat_adjust=True)
        ptop (in): The pressure level at the top of atmosphere
        akap (in): Poisson constant (KAPPA)
        zvir (in): Constant (Rv/Rd-1)
        last_step (in): Flag for the last step of k-split remapping
        consv_te (in): If True, conserve total energy
        mdt (in) : Remap time step
        bdt (in): Timestep
        do_adiabatic_init (in): If True, do adiabatic dynamics

        Remap the deformed Lagrangian surfaces onto the reference, or "Eulerian",
        coordinate levels.
        """
        # TODO: remove unused arguments (and commented code that references them)
        # TODO: can we trim ps or make it a temporary
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

        self._pn2_pk_delp(self._dp2, delp, self._pe2, self._pn2, pk, akap)

        self._map_single_pt(pt, peln, self._pn2, qmin=self._t_min)

        # TODO if self._nq > 5:
        self._mapn_tracer(self._pe1, self._pe2, self._dp2, tracers)
        # TODO else if self._nq > 0:
        # TODO map1_q2, fillz

        self._map_single_w(w, self._pe1, self._pe2, qs=wsd)
        self._map_single_delz(delz, self._pe1, self._pe2)

        self._undo_delz_adjust_and_copy_peln(delp, delz, peln, self._pe0, self._pn2)
        # if do_omega:  # NOTE untested
        #    pe3 = copy(omga, origin=(grid_indexing.isc, grid_indexing.jsc, 1))

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
            if consv_te > CONSV_MIN:
                raise NotImplementedError(
                    "We do not support consv_te > 0.001 "
                    "because that would trigger an allReduce"
                )
            elif consv_te < -CONSV_MIN:
                raise NotImplementedError(
                    "Unimplemented/untested case consv("
                    + str(consv_te)
                    + ")  < -CONSV_MIN("
                    + str(-CONSV_MIN)
                    + ")"
                )

        if self._do_sat_adjust:
            fast_mp_consv = not do_adiabatic_init and consv_te > CONSV_MIN
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
