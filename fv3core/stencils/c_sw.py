import gt4py.gtscript as gtscript
from gt4py.gtscript import __INLINED, PARALLEL, computation, interval, parallel, region

import fv3core._config as spec
import fv3core.stencils.d2a2c_vect as d2a2c
import fv3core.stencils.vorticitytransport_cgrid as vorticity_transport
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.utils.corners import fill_4corners_x, fill_4corners_y


sd = utils.sd


@gtstencil()
def geoadjust_ut(ut: sd, dy: sd, sin_sg3: sd, sin_sg1: sd, dt2: float):
    with computation(PARALLEL), interval(...):
        ut[0, 0, 0] = (
            dt2 * ut * dy * sin_sg3[-1, 0, 0] if ut > 0 else dt2 * ut * dy * sin_sg1
        )


@gtstencil()
def geoadjust_vt(vt: sd, dx: sd, sin_sg4: sd, sin_sg2: sd, dt2: float):
    with computation(PARALLEL), interval(...):
        vt[0, 0, 0] = (
            dt2 * vt * dx * sin_sg4[0, -1, 0] if vt > 0 else dt2 * vt * dx * sin_sg2
        )


@gtstencil()
def absolute_vorticity(vort: sd, fC: sd, rarea_c: sd):
    with computation(PARALLEL), interval(...):
        vort[0, 0, 0] = fC + rarea_c * vort


@gtscript.function
def nonhydro_x_fluxes(delp: sd, pt: sd, w: sd, utc: sd):
    fx1 = delp[-1, 0, 0] if utc > 0.0 else delp
    fx = pt[-1, 0, 0] if utc > 0.0 else pt
    fx2 = w[-1, 0, 0] if utc > 0.0 else w
    fx1 = utc * fx1
    fx = fx1 * fx
    fx2 = fx1 * fx2
    return fx, fx1, fx2


@gtscript.function
def nonhydro_y_fluxes(delp: sd, pt: sd, w: sd, vtc: sd):
    fy1 = delp[0, -1, 0] if vtc > 0.0 else delp
    fy = pt[0, -1, 0] if vtc > 0.0 else pt
    fy2 = w[0, -1, 0] if vtc > 0.0 else w
    fy1 = vtc * fy1
    fy = fy1 * fy
    fy2 = fy1 * fy2
    return fy, fy1, fy2


@gtstencil()
def transportdelp(
    delp: sd, pt: sd, utc: sd, vtc: sd, w: sd, rarea: sd, delpc: sd, ptc: sd, wc: sd
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
        assert __INLINED(namelist.grid_type < 3)
        # additional assumption (not grid.nested)

        delp = fill_4corners_x(delp)
        pt = fill_4corners_x(pt)
        w = fill_4corners_x(w)

        fx, fx1, fx2 = nonhydro_x_fluxes(delp, pt, w, utc)

        delp = fill_4corners_y(delp)
        pt = fill_4corners_y(pt)
        w = fill_4corners_y(w)

        fy, fy1, fy2 = nonhydro_y_fluxes(delp, pt, w, vtc)

        delpc = delp + (fx1 - fx1[1, 0, 0] + fy1 - fy1[0, 1, 0]) * rarea
        ptc = (pt * delp + (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea) / delpc
        wc = (w * delp + (fx2 - fx2[1, 0, 0] + fy2 - fy2[0, 1, 0]) * rarea) / delpc


@gtstencil()
def divergence_corner(
    u: sd,
    v: sd,
    ua: sd,
    va: sd,
    dxc: sd,
    dyc: sd,
    sin_sg1: sd,
    sin_sg2: sd,
    sin_sg3: sd,
    sin_sg4: sd,
    cos_sg1: sd,
    cos_sg2: sd,
    cos_sg3: sd,
    cos_sg4: sd,
    rarea_c: sd,
    divg_d: sd,
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
            (u - 0.25 * (va[0, -1, 0] + va) * (cos_sg4[0, -1, 0] + cos_sg2))
            * dyc
            * 0.5
            * (sin_sg4[0, -1, 0] + sin_sg2)
        )
        with parallel(region[:, j_start], region[:, j_end + 1]):
            uf = u * dyc * 0.5 * (sin_sg4[0, -1, 0] + sin_sg2)

        vf = (
            (v - 0.25 * (ua[-1, 0, 0] + ua) * (cos_sg3[-1, 0, 0] + cos_sg1))
            * dxc
            * 0.5
            * (sin_sg3[-1, 0, 0] + sin_sg1)
        )
        with parallel(region[i_start, :], region[i_end + 1, :]):
            vf = v * dxc * 0.5 * (sin_sg3[-1, 0, 0] + sin_sg1)

        divg_d = vf[0, -1, 0] - vf + uf[-1, 0, 0] - uf
        with parallel(region[i_start, j_start], region[i_end + 1, j_start]):
            divg_d -= vf[0, -1, 0]
        with parallel(region[i_end + 1, j_end + 1], region[i_start, j_end + 1]):
            divg_d += vf
        divg_d *= rarea_c


@gtstencil()
def circulation_cgrid(uc: sd, vc: sd, dxc: sd, dyc: sd, vort_c: sd):
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

        with parallel(region[i_start, j_start], region[i_start, j_end + 1]):
            vort_c += fy[-1, 0, 0]

        with parallel(region[i_end + 1, j_start], region[i_end + 1, j_end + 1]):
            vort_c -= fy[0, 0, 0]


@gtstencil()
def update_vorticity_and_kinetic_energy(
    ke: sd,
    vort: sd,
    ua: sd,
    va: sd,
    uc: sd,
    vc: sd,
    u: sd,
    v: sd,
    sin_sg1: sd,
    cos_sg1: sd,
    sin_sg2: sd,
    cos_sg2: sd,
    sin_sg3: sd,
    cos_sg3: sd,
    sin_sg4: sd,
    cos_sg4: sd,
    dt2: float,
):
    from __externals__ import i_end, i_start, j_end, j_start, namelist

    with computation(PARALLEL), interval(...):
        assert __INLINED(namelist.grid_type < 3)

        ke = uc if ua > 0.0 else uc[1, 0, 0]
        vort = vc if va > 0.0 else vc[0, 1, 0]

        with parallel(region[:, j_start - 1], region[:, j_end]):
            vort = vort * sin_sg4 + u[0, 1, 0] * cos_sg4 if va <= 0.0 else vort
        with parallel(region[:, j_start], region[:, j_end + 1]):
            vort = vort * sin_sg2 + u * cos_sg2 if va > 0.0 else vort

        with parallel(region[i_end, :], region[i_start - 1, :]):
            ke = ke * sin_sg3 + v[1, 0, 0] * cos_sg3 if ua <= 0.0 else ke
        with parallel(region[i_end + 1, :], region[i_start, :]):
            ke = ke * sin_sg1 + v * cos_sg1 if ua > 0.0 else ke

        ke = 0.5 * dt2 * (ua * ke + va * vort)


def compute(delp, pt, u, v, w, uc, vc, ua, va, ut, vt, divgd, omga, dt2):
    grid = spec.grid
    dord4 = True
    origin_halo1 = (grid.is_ - 1, grid.js - 1, 0)
    delpc = utils.make_storage_from_shape(delp.shape, origin=origin_halo1)
    ptc = utils.make_storage_from_shape(pt.shape, origin=origin_halo1)
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
            domain=grid.domain_shape_compute_buffer_2d(add=(1, 1, 0)),
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
    ke = utils.make_storage_from_shape(uc.shape)
    vort = utils.make_storage_from_shape(vc.shape)

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
        domain=grid.domain_shape_compute_buffer_2d(add=(1, 1, 0)),
    )
    absolute_vorticity(
        vort,
        grid.fC,
        grid.rarea_c,
        origin=grid.compute_origin(),
        domain=(grid.nic + 1, grid.njc + 1, grid.npz),
    )
    vorticity_transport.compute(uc, vc, vort, ke, v, u, dt2)
    return delpc, ptc
