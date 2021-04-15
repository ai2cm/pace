import gt4py.gtscript as gtscript
from gt4py.gtscript import (
    BACKWARD,
    FORWARD,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)

import fv3core._config as spec
from fv3core.decorators import gtstencil
from fv3core.stencils.rayleigh_super import SDAY, compute_rf_vals
from fv3core.utils.typing import FloatField, FloatFieldK


@gtscript.function
def compute_rff_vals(pfull, dt, rf_cutoff, tau0, ptop):
    rffvals = compute_rf_vals(pfull, dt, rf_cutoff, tau0, ptop)
    rffvals = 1.0 / (1.0 + rffvals)
    return rffvals


@gtscript.function
def dm_layer(rf, dp, wind):
    return (1.0 - rf) * dp * wind


@gtstencil()
def ray_fast_wind(
    u: FloatField,
    v: FloatField,
    w: FloatField,
    dp: FloatFieldK,
    pfull: FloatFieldK,
    dt: float,
    ptop: float,
    rf_cutoff_nudge: float,
    ks: int,
    hydrostatic: bool,
):
    from __externals__ import local_ie, local_je, namelist

    # dm_stencil
    with computation(PARALLEL), interval(...):
        # TODO -- in the fortran model rf is only computed once, repeating
        # the computation every time ray_fast is run is inefficient
        if pfull < namelist.rf_cutoff:
            rf = compute_rff_vals(
                pfull, dt, namelist.rf_cutoff, namelist.tau * SDAY, ptop
            )
    with computation(FORWARD):
        with interval(0, 1):
            if pfull < rf_cutoff_nudge:  # TODO and kaxes(k) < ks:
                dm = dp
        with interval(1, None):
            dm = dm[0, 0, -1]
            if pfull < rf_cutoff_nudge:  # TODO and kaxes(k) < ks:
                dm += dp
    with computation(BACKWARD), interval(0, -1):
        if pfull < rf_cutoff_nudge:
            dm = dm[0, 0, 1]
    # ray_fast_wind(u)
    with computation(FORWARD):
        with interval(0, 1):
            with horizontal(region[: local_ie + 1, :]):
                if pfull < namelist.rf_cutoff:
                    dmdir = dm_layer(rf, dp, u)
                    u *= rf
                else:
                    dm = 0
        with interval(1, None):
            with horizontal(region[: local_ie + 1, :]):
                dmdir = dmdir[0, 0, -1]
                if pfull < namelist.rf_cutoff:
                    dmdir += dm_layer(rf, dp, u)
                    u *= rf
    with computation(BACKWARD), interval(0, -1):
        if pfull < namelist.rf_cutoff:
            dmdir = dmdir[0, 0, 1]
    with computation(PARALLEL), interval(...):
        with horizontal(region[: local_ie + 1, :]):
            if pfull < rf_cutoff_nudge:  # TODO and axes(k) < ks:
                u += dmdir / dm
    # ray_fast_wind(v)
    with computation(FORWARD):
        with interval(0, 1):
            with horizontal(region[:, : local_je + 1]):
                if pfull < namelist.rf_cutoff:
                    dmdir = dm_layer(rf, dp, v)
                    v *= rf
                else:
                    dm = 0
        with interval(1, None):
            with horizontal(region[:, : local_je + 1]):
                dmdir = dmdir[0, 0, -1]
                if pfull < namelist.rf_cutoff:
                    dmdir += dm_layer(rf, dp, v)
                    v *= rf
    with computation(BACKWARD), interval(0, -1):
        if pfull < namelist.rf_cutoff:
            dmdir = dmdir[0, 0, 1]
    with computation(PARALLEL), interval(...):
        with horizontal(region[:, : local_je + 1]):
            if pfull < rf_cutoff_nudge:  # TODO and axes(k) < ks:
                v += dmdir / dm
    # ray_fast_w
    with computation(PARALLEL), interval(...):
        with horizontal(region[: local_ie + 1, : local_je + 1]):
            if not hydrostatic and pfull < namelist.rf_cutoff:
                w *= rf


def compute(
    u: FloatField,
    v: FloatField,
    w: FloatField,
    dp: FloatFieldK,
    pfull: FloatFieldK,
    dt: float,
    ptop: float,
    ks: int,
):
    grid = spec.grid
    rf_cutoff_nudge = spec.namelist.rf_cutoff + min(100.0, 10.0 * ptop)

    ray_fast_wind(
        u,
        v,
        w,
        dp,
        pfull,
        dt,
        ptop,
        rf_cutoff_nudge,
        ks,
        hydrostatic=spec.namelist.hydrostatic,
        origin=grid.compute_origin(),
        domain=(grid.nic + 1, grid.njc + 1, grid.npz),
    )
