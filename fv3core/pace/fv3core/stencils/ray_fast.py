import gt4py.cartesian.gtscript as gtscript
from gt4py.cartesian.gtscript import (
    __INLINED,
    BACKWARD,
    FORWARD,
    PARALLEL,
    computation,
    horizontal,
    interval,
    log,
    region,
    sin,
)

import pace.util.constants as constants
from pace.dsl.dace.orchestration import orchestrate
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldK
from pace.util import X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM


SDAY = 86400.0

# NOTE: The fortran version of this computes rf in the first timestep only. Then
# rf_initialized let's you know you can skip it. Here we calculate it every
# time.
@gtscript.function
def compute_rf_vals(pfull, bdt, rf_cutoff, tau0, ptop):
    return (
        bdt
        / tau0
        * sin(0.5 * constants.PI * log(rf_cutoff / pfull) / log(rf_cutoff / ptop)) ** 2
    )


@gtscript.function
def compute_rff_vals(pfull, dt, rf_cutoff, tau0, ptop):
    rffvals = compute_rf_vals(pfull, dt, rf_cutoff, tau0, ptop)
    rffvals = 1.0 / (1.0 + rffvals)
    return rffvals


@gtscript.function
def dm_layer(rf, dp, wind):
    return (1.0 - rf) * dp * wind


def ray_fast_wind_compute(
    u: FloatField,
    v: FloatField,
    w: FloatField,
    delta_p_ref: FloatFieldK,  # reference delta pressure
    pfull: FloatFieldK,  # input layer pressure reference?
    dt: float,
    ptop: float,
    rf_cutoff_nudge: float,
):
    """
    Args:
        u (inout):
        v (inout):
        w (inout):
        delta_p_ref (in):
        pfull (in):
        dt (in):
        ptop (in):
        rf_cutoff_nudge (in):
        ks (in):
    """
    from __externals__ import hydrostatic, local_ie, local_je, rf_cutoff, tau

    # dm_stencil
    with computation(PARALLEL), interval(...):
        # TODO -- in the fortran model rf is only computed once, repeating
        # the computation every time ray_fast is run is inefficient
        if pfull < rf_cutoff:
            # rf is rayleigh damping increment, fraction of vertical velocity
            # left after doing rayleigh damping (w -> w * rf)
            rf = compute_rff_vals(pfull, dt, rf_cutoff, tau * SDAY, ptop)
    with computation(FORWARD):
        with interval(0, 1):
            if pfull < rf_cutoff_nudge:
                p_ref = delta_p_ref
        with interval(1, None):
            p_ref = p_ref[0, 0, -1]
            if pfull < rf_cutoff_nudge:
                p_ref += delta_p_ref
    with computation(BACKWARD), interval(0, -1):
        if pfull < rf_cutoff_nudge:
            p_ref = p_ref[0, 0, 1]
    # ray_fast_wind(u)
    with computation(FORWARD):
        with interval(0, 1):
            with horizontal(region[: local_ie + 1, :]):
                if pfull < rf_cutoff:
                    # dmdir = (1.0 - rf) * dp * wind
                    dmdir = dm_layer(rf, delta_p_ref, u)
                    u *= rf
                else:
                    p_ref = 0
        with interval(1, None):
            with horizontal(region[: local_ie + 1, :]):
                dmdir = dmdir[0, 0, -1]
                if pfull < rf_cutoff:
                    dmdir += dm_layer(rf, delta_p_ref, u)
                    u *= rf
    with computation(BACKWARD), interval(0, -1):
        if pfull < rf_cutoff:
            dmdir = dmdir[0, 0, 1]
    with computation(PARALLEL), interval(...):
        with horizontal(region[: local_ie + 1, :]):
            if pfull < rf_cutoff_nudge:
                u += dmdir / p_ref
    # ray_fast_wind(v)
    with computation(FORWARD):
        with interval(0, 1):
            with horizontal(region[:, : local_je + 1]):
                if pfull < rf_cutoff:
                    dmdir = dm_layer(rf, delta_p_ref, v)
                    v *= rf
                else:
                    p_ref = 0
        with interval(1, None):
            with horizontal(region[:, : local_je + 1]):
                dmdir = dmdir[0, 0, -1]
                if pfull < rf_cutoff:
                    dmdir += dm_layer(rf, delta_p_ref, v)
                    v *= rf
    with computation(BACKWARD), interval(0, -1):
        if pfull < rf_cutoff:
            dmdir = dmdir[0, 0, 1]
    with computation(PARALLEL), interval(...):
        with horizontal(region[:, : local_je + 1]):
            if pfull < rf_cutoff_nudge:
                v += dmdir / p_ref
    # ray_fast_w
    with computation(PARALLEL), interval(...):
        with horizontal(region[: local_ie + 1, : local_je + 1]):
            if __INLINED(not hydrostatic):
                if pfull < rf_cutoff:
                    w *= rf


class RayleighDamping:
    """
    Apply Rayleigh damping (for tau > 0).

    Namelist:
        - tau [Float]: time scale (in days) for Rayleigh friction applied to horizontal
                       and vertical winds; lost kinetic energy is converted to heat,
                       except on nested grids.
        - rf_cutoff [Float]: pressure below which no Rayleigh damping is applied
                             if tau > 0.

    Fortran name: ray_fast.
    """

    def __init__(self, stencil_factory: StencilFactory, rf_cutoff, tau, hydrostatic):
        orchestrate(obj=self, config=stencil_factory.config.dace_config)
        grid_indexing = stencil_factory.grid_indexing
        self._rf_cutoff = rf_cutoff
        origin, domain = grid_indexing.get_origin_domain(
            [X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM]
        )

        ax_offsets = grid_indexing.axis_offsets(origin, domain)
        local_axis_offsets = {}
        for axis_offset_name, axis_offset_value in ax_offsets.items():
            if "local" in axis_offset_name:
                local_axis_offsets[axis_offset_name] = axis_offset_value

        self._ray_fast_wind_compute = stencil_factory.from_origin_domain(
            ray_fast_wind_compute,
            origin=origin,
            domain=domain,
            externals={
                "hydrostatic": hydrostatic,
                "rf_cutoff": rf_cutoff,
                "tau": tau,
                **local_axis_offsets,
            },
        )

    def __call__(
        self,
        u: FloatField,
        v: FloatField,
        w: FloatField,
        dp: FloatFieldK,
        pfull: FloatFieldK,
        dt: float,
        ptop: float,
    ):

        rf_cutoff_nudge = self._rf_cutoff + min(100.0, 10.0 * ptop)

        self._ray_fast_wind_compute(
            u,
            v,
            w,
            dp,
            pfull,
            dt,
            ptop,
            rf_cutoff_nudge,
        )
