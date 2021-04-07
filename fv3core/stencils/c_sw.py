import gt4py.gtscript as gtscript
from gt4py.gtscript import (
    PARALLEL,
    computation,
    external_assert,
    horizontal,
    interval,
    region,
)

import fv3core._config as spec
import fv3core.stencils.d2a2c_vect as d2a2c
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.utils import corners
from fv3core.utils.typing import FloatField, FloatFieldIJ


@gtstencil()
def geoadjust_ut(
    ut: FloatField,
    dy: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    sin_sg1: FloatFieldIJ,
    dt2: float,
):
    with computation(PARALLEL), interval(...):
        ut[0, 0, 0] = (
            dt2 * ut * dy * sin_sg3[-1, 0] if ut > 0 else dt2 * ut * dy * sin_sg1
        )


@gtstencil()
def geoadjust_vt(
    vt: FloatField,
    dx: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    sin_sg2: FloatFieldIJ,
    dt2: float,
):
    with computation(PARALLEL), interval(...):
        vt[0, 0, 0] = (
            dt2 * vt * dx * sin_sg4[0, -1] if vt > 0 else dt2 * vt * dx * sin_sg2
        )


@gtstencil()
def absolute_vorticity(vort: FloatField, fC: FloatFieldIJ, rarea_c: FloatFieldIJ):
    with computation(PARALLEL), interval(...):
        vort[0, 0, 0] = fC + rarea_c * vort


@gtscript.function
def nonhydro_x_fluxes(delp: FloatField, pt: FloatField, w: FloatField, utc: FloatField):
    fx1 = delp[-1, 0, 0] if utc > 0.0 else delp
    fx = pt[-1, 0, 0] if utc > 0.0 else pt
    fx2 = w[-1, 0, 0] if utc > 0.0 else w
    fx1 = utc * fx1
    fx = fx1 * fx
    fx2 = fx1 * fx2
    return fx, fx1, fx2


@gtscript.function
def nonhydro_y_fluxes(delp: FloatField, pt: FloatField, w: FloatField, vtc: FloatField):
    fy1 = delp[0, -1, 0] if vtc > 0.0 else delp
    fy = pt[0, -1, 0] if vtc > 0.0 else pt
    fy2 = w[0, -1, 0] if vtc > 0.0 else w
    fy1 = vtc * fy1
    fy = fy1 * fy
    fy2 = fy1 * fy2
    return fy, fy1, fy2


@gtstencil()
def transportdelp(
    delp: FloatField,
    pt: FloatField,
    utc: FloatField,
    vtc: FloatField,
    w: FloatField,
    rarea: FloatFieldIJ,
    delpc: FloatField,
    ptc: FloatField,
    wc: FloatField,
):
    """Transport delp.

    Args:
        delp: What is transported (input)
        pt: Pressure (input)
        utc: x-velocity on C-grid (input)
        vtc: y-velocity on C-grid (input)
        w: z-velocity on C-grid (input)
        rarea: Inverse areas (input) -- IJ field
        delpc: Updated delp (output)
        ptc: Updated pt (output)
        wc: Updated w (output)
    """

    from __externals__ import namelist

    with computation(PARALLEL), interval(...):
        external_assert(namelist.grid_type < 3)
        # additional assumption (not grid.nested)

        delp = corners.fill_corners_2cells_x(delp)
        pt = corners.fill_corners_2cells_x(pt)
        w = corners.fill_corners_2cells_x(w)

        fx, fx1, fx2 = nonhydro_x_fluxes(delp, pt, w, utc)

        delp = corners.fill_corners_2cells_y(delp)
        pt = corners.fill_corners_2cells_y(pt)
        w = corners.fill_corners_2cells_y(w)

        fy, fy1, fy2 = nonhydro_y_fluxes(delp, pt, w, vtc)

        delpc = delp + (fx1 - fx1[1, 0, 0] + fy1 - fy1[0, 1, 0]) * rarea
        ptc = (pt * delp + (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea) / delpc
        wc = (w * delp + (fx2 - fx2[1, 0, 0] + fy2 - fy2[0, 1, 0]) * rarea) / delpc


@gtstencil()
def divergence_corner(
    u: FloatField,
    v: FloatField,
    ua: FloatField,
    va: FloatField,
    dxc: FloatFieldIJ,
    dyc: FloatFieldIJ,
    sin_sg1: FloatFieldIJ,
    sin_sg2: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    cos_sg1: FloatFieldIJ,
    cos_sg2: FloatFieldIJ,
    cos_sg3: FloatFieldIJ,
    cos_sg4: FloatFieldIJ,
    rarea_c: FloatFieldIJ,
    divg_d: FloatField,
):
    """Calculate divg on d-grid.
    Args:
        u: x-velocity (input)
        v: y-velocity (input)
        ua: x-velocity on a (input)
        va: y-velocity on a (input)
        dxc: grid spacing in x-direction (input)
        dyc: grid spacing in y-direction (input)
        sin_sg1: grid sin(sg1) (input)
        sin_sg2: grid sin(sg2) (input)
        sin_sg3: grid sin(sg3) (input)
        sin_sg4: grid sin(sg4) (input)
        cos_sg1: grid cos(sg1) (input)
        cos_sg2: grid cos(sg2) (input)
        cos_sg3: grid cos(sg3) (input)
        cos_sg4: grid cos(sg4) (input)
        rarea_c: inverse cell areas on c-grid (input)
        divg_d: divergence on d-grid (output)
    """
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        uf = (
            (u - 0.25 * (va[0, -1, 0] + va) * (cos_sg4[0, -1] + cos_sg2))
            * dyc
            * 0.5
            * (sin_sg4[0, -1] + sin_sg2)
        )
        with horizontal(region[:, j_start], region[:, j_end + 1]):
            uf = u * dyc * 0.5 * (sin_sg4[0, -1] + sin_sg2)

        vf = (
            (v - 0.25 * (ua[-1, 0, 0] + ua) * (cos_sg3[-1, 0] + cos_sg1))
            * dxc
            * 0.5
            * (sin_sg3[-1, 0] + sin_sg1)
        )
        with horizontal(region[i_start, :], region[i_end + 1, :]):
            vf = v * dxc * 0.5 * (sin_sg3[-1, 0] + sin_sg1)

        divg_d = vf[0, -1, 0] - vf + uf[-1, 0, 0] - uf
        with horizontal(region[i_start, j_start], region[i_end + 1, j_start]):
            divg_d -= vf[0, -1, 0]
        with horizontal(region[i_end + 1, j_end + 1], region[i_start, j_end + 1]):
            divg_d += vf
        divg_d *= rarea_c


@gtstencil()
def circulation_cgrid(
    uc: FloatField,
    vc: FloatField,
    dxc: FloatFieldIJ,
    dyc: FloatFieldIJ,
    vort_c: FloatField,
):
    """Update vort_c.

    Args:
        uc: x-velocity on C-grid (input)
        vc: y-velocity on C-grid (input)
        dxc: grid spacing in x-dir (input)
        dyc: grid spacing in y-dir (input)
        vort_c: C-grid vorticity (output)
    """
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        fx = dxc * uc
        fy = dyc * vc

        vort_c = fx[0, -1, 0] - fx - fy[-1, 0, 0] + fy

        with horizontal(region[i_start, j_start], region[i_start, j_end + 1]):
            vort_c += fy[-1, 0, 0]

        with horizontal(region[i_end + 1, j_start], region[i_end + 1, j_end + 1]):
            vort_c -= fy[0, 0, 0]


@gtstencil()
def update_vorticity_and_kinetic_energy(
    ke: FloatField,
    vort: FloatField,
    ua: FloatField,
    va: FloatField,
    uc: FloatField,
    vc: FloatField,
    u: FloatField,
    v: FloatField,
    sin_sg1: FloatFieldIJ,
    cos_sg1: FloatFieldIJ,
    sin_sg2: FloatFieldIJ,
    cos_sg2: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    cos_sg3: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    cos_sg4: FloatFieldIJ,
    dt2: float,
):
    from __externals__ import i_end, i_start, j_end, j_start, namelist

    with computation(PARALLEL), interval(...):
        external_assert(namelist.grid_type < 3)

        ke = uc if ua > 0.0 else uc[1, 0, 0]
        vort = vc if va > 0.0 else vc[0, 1, 0]

        with horizontal(region[:, j_start - 1], region[:, j_end]):
            vort = vort * sin_sg4 + u[0, 1, 0] * cos_sg4 if va <= 0.0 else vort
        with horizontal(region[:, j_start], region[:, j_end + 1]):
            vort = vort * sin_sg2 + u * cos_sg2 if va > 0.0 else vort

        with horizontal(region[i_end, :], region[i_start - 1, :]):
            ke = ke * sin_sg3 + v[1, 0, 0] * cos_sg3 if ua <= 0.0 else ke
        with horizontal(region[i_end + 1, :], region[i_start, :]):
            ke = ke * sin_sg1 + v * cos_sg1 if ua > 0.0 else ke

        ke = 0.5 * dt2 * (ua * ke + va * vort)


@gtstencil()
def update_zonal_velocity(
    vorticity: FloatField,
    ke: FloatField,
    velocity: FloatField,
    velocity_c: FloatField,
    cosa: FloatFieldIJ,
    sina: FloatFieldIJ,
    rdxc: FloatFieldIJ,
    dt2: float,
):
    from __externals__ import i_end, i_start, namelist

    with computation(PARALLEL), interval(...):
        external_assert(namelist.grid_type < 3)
        # additional assumption: not __INLINED(spec.grid.nested)

        tmp_flux = dt2 * (velocity - velocity_c * cosa) / sina
        with horizontal(region[i_start, :], region[i_end + 1, :]):
            tmp_flux = dt2 * velocity

        flux = vorticity[0, 0, 0] if tmp_flux > 0.0 else vorticity[0, 1, 0]
        velocity_c = velocity_c + tmp_flux * flux + rdxc * (ke[-1, 0, 0] - ke)


@gtstencil()
def update_meridional_velocity(
    vorticity: FloatField,
    ke: FloatField,
    velocity: FloatField,
    velocity_c: FloatField,
    cosa: FloatFieldIJ,
    sina: FloatFieldIJ,
    rdyc: FloatFieldIJ,
    dt2: float,
):
    from __externals__ import j_end, j_start, namelist

    with computation(PARALLEL), interval(...):
        external_assert(namelist.grid_type < 3)
        # additional assumption: not __INLINED(spec.grid.nested)

        tmp_flux = dt2 * (velocity - velocity_c * cosa) / sina
        with horizontal(region[:, j_start], region[:, j_end + 1]):
            tmp_flux = dt2 * velocity

        flux = vorticity[0, 0, 0] if tmp_flux > 0.0 else vorticity[1, 0, 0]
        velocity_c = velocity_c - tmp_flux * flux + rdyc * (ke[0, -1, 0] - ke)


@gtstencil
def initialize_delpc_ptc(delpc: FloatField, ptc: FloatField):
    with computation(PARALLEL), interval(...):
        delpc = 0.0
        ptc = 0.0


def vorticitytransport_cgrid(
    uc: FloatField,
    vc: FloatField,
    vort_c: FloatField,
    ke_c: FloatField,
    v: FloatField,
    u: FloatField,
    dt2: float,
):
    """Update the C-Grid zonal and meridional velocity fields.

    Args:
        uc: x-velocity on C-grid (input, output)
        vc: y-velocity on C-grid (input, output)
        vort_c: Vorticity on C-grid (input)
        ke_c: kinetic energy on C-grid (input)
        v: y-velocity on D-grid (input)
        u: x-velocity on D-grid (input)
        dt2: timestep (input)
    """
    grid = spec.grid
    update_meridional_velocity(
        vort_c,
        ke_c,
        u,
        vc,
        grid.cosa_v,
        grid.sina_v,
        grid.rdyc,
        dt2,
        origin=grid.compute_origin(),
        domain=grid.domain_shape_compute(add=(0, 1, 0)),
    )
    update_zonal_velocity(
        vort_c,
        ke_c,
        v,
        uc,
        grid.cosa_u,
        grid.sina_u,
        grid.rdxc,
        dt2,
        origin=grid.compute_origin(),
        domain=grid.domain_shape_compute(add=(1, 0, 0)),
    )


def compute(delp, pt, u, v, w, uc, vc, ua, va, ut, vt, divgd, omga, dt2):
    grid = spec.grid
    dord4 = True
    origin_halo1 = (grid.is_ - 1, grid.js - 1, 0)
    delpc = utils.make_storage_from_shape(
        delp.shape, origin=origin_halo1, cache_key="c_sw_delpc"
    )
    ptc = utils.make_storage_from_shape(
        pt.shape, origin=origin_halo1, cache_key="c_sw_ptc"
    )
    initialize_delpc_ptc(
        delpc, ptc, origin=grid.full_origin(), domain=grid.domain_shape_full()
    )
    d2a2c.compute(dord4, uc, vc, u, v, ua, va, ut, vt)
    if spec.namelist.nord > 0:
        divergence_corner(
            u,
            v,
            ua,
            va,
            grid.dxc,
            grid.dyc,
            grid.sin_sg1,
            grid.sin_sg2,
            grid.sin_sg3,
            grid.sin_sg4,
            grid.cos_sg1,
            grid.cos_sg2,
            grid.cos_sg3,
            grid.cos_sg4,
            grid.rarea_c,
            divgd,
            origin=grid.compute_origin(),
            domain=grid.domain_shape_compute(add=(1, 1, 0)),
        )
    geo_origin = (grid.is_ - 1, grid.js - 1, 0)
    geoadjust_ut(
        ut,
        grid.dy,
        grid.sin_sg3,
        grid.sin_sg1,
        dt2,
        origin=geo_origin,
        domain=(grid.nic + 3, grid.njc + 2, grid.npz),
    )
    geoadjust_vt(
        vt,
        grid.dx,
        grid.sin_sg4,
        grid.sin_sg2,
        dt2,
        origin=geo_origin,
        domain=(grid.nic + 2, grid.njc + 3, grid.npz),
    )
    transportdelp(
        delp,
        pt,
        ut,
        vt,
        w,
        grid.rarea,
        delpc,
        ptc,
        omga,
        origin=geo_origin,
        domain=(grid.nic + 2, grid.njc + 2, grid.npz),
    )

    # ke_c_sw
    # {
    # Create storage objects to hold the new vorticity and kinetic energy values
    ke = utils.make_storage_from_shape(uc.shape, cache_key="c_sw_ke")
    vort = utils.make_storage_from_shape(vc.shape, cache_key="c_sw_vort")

    # Set vorticity and kinetic energy values
    update_vorticity_and_kinetic_energy(
        ke,
        vort,
        ua,
        va,
        uc,
        vc,
        u,
        v,
        grid.sin_sg1,
        grid.cos_sg1,
        grid.sin_sg2,
        grid.cos_sg2,
        grid.sin_sg3,
        grid.cos_sg3,
        grid.sin_sg4,
        grid.cos_sg4,
        dt2,
        origin=(grid.is_ - 1, grid.js - 1, 0),
        domain=(grid.nic + 2, grid.njc + 2, grid.npz),
    )
    # }

    # ke, vort = ke_c_sw.compute(uc, vc, u, v, ua, va, dt2)
    circulation_cgrid(
        uc,
        vc,
        grid.dxc,
        grid.dyc,
        vort,
        origin=grid.compute_origin(),
        domain=grid.domain_shape_compute(add=(1, 1, 0)),
    )
    absolute_vorticity(
        vort,
        grid.fC,
        grid.rarea_c,
        origin=grid.compute_origin(),
        domain=(grid.nic + 1, grid.njc + 1, grid.npz),
    )
    vorticitytransport_cgrid(uc, vc, vort, ke, v, u, dt2)
    return delpc, ptc
