from gt4py import gtscript
from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.d2a2c_vect import d2a2c_vect
from fv3core.utils import corners
from fv3core.utils.typing import FloatField


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


@gtscript.function
def transportdelp(
    delp: FloatField,
    pt: FloatField,
    utc: FloatField,
    vtc: FloatField,
    w: FloatField,
    rarea: FloatField,
):
    """Transport delp.

    Args:
        delp: What is transported
        pt: Pressure
        utc: x-velocity on C-grid
        vtc: y-velocity on C-grid
        w: z-velocity on C-grid
        rarea: Inverse areas -- IJ field

    Returns:
        delpc: Updated delp
        ptc: Updated pt
        wc: Updated w
    """

    from __externals__ import namelist

    assert __INLINED(namelist.grid_type < 3)
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

    return delpc, ptc, wc


@gtscript.function
def divergence_corner(
    u: FloatField,
    v: FloatField,
    ua: FloatField,
    va: FloatField,
    dxc: FloatField,
    dyc: FloatField,
    sin_sg1: FloatField,
    sin_sg2: FloatField,
    sin_sg3: FloatField,
    sin_sg4: FloatField,
    cos_sg1: FloatField,
    cos_sg2: FloatField,
    cos_sg3: FloatField,
    cos_sg4: FloatField,
    rarea_c: FloatField,
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

    Returns:
        divg_d: divergence on d-grid (output)
    """
    from __externals__ import i_end, i_start, j_end, j_start, namelist

    if __INLINED(namelist.nord > 0):
        uf = (
            (u - 0.25 * (va[0, -1, 0] + va) * (cos_sg4[0, -1, 0] + cos_sg2))
            * dyc
            * 0.5
            * (sin_sg4[0, -1, 0] + sin_sg2)
        )
        with horizontal(region[:, j_start], region[:, j_end + 1]):
            uf = u * dyc * 0.5 * (sin_sg4[0, -1, 0] + sin_sg2)

        vf = (
            (v - 0.25 * (ua[-1, 0, 0] + ua) * (cos_sg3[-1, 0, 0] + cos_sg1))
            * dxc
            * 0.5
            * (sin_sg3[-1, 0, 0] + sin_sg1)
        )
        with horizontal(region[i_start, :], region[i_end + 1, :]):
            vf = v * dxc * 0.5 * (sin_sg3[-1, 0, 0] + sin_sg1)

        divg_d = vf[0, -1, 0] - vf + uf[-1, 0, 0] - uf
        with horizontal(region[i_start, j_start], region[i_end + 1, j_start]):
            divg_d -= vf[0, -1, 0]
        with horizontal(region[i_end + 1, j_end + 1], region[i_start, j_end + 1]):
            divg_d += vf
        divg_d *= rarea_c

    else:
        divg_d = 0.0

    return divg_d


@gtscript.function
def circulation_cgrid(uc: FloatField, vc: FloatField, dxc: FloatField, dyc: FloatField):
    """Update vort_c.

    Args:
        uc: x-velocity on C-grid (input)
        vc: y-velocity on C-grid (input)
        dxc: grid spacing in x-dir (input)
        dyc: grid spacing in y-dir (input)

    Returns:
        vort_c: C-grid vorticity (output)
    """
    from __externals__ import i_end, i_start, j_end, j_start

    fx = dxc * uc
    fy = dyc * vc

    vort_c = fx[0, -1, 0] - fx - fy[-1, 0, 0] + fy

    with horizontal(region[i_start, j_start], region[i_start, j_end + 1]):
        vort_c += fy[-1, 0, 0]

    with horizontal(region[i_end + 1, j_start], region[i_end + 1, j_end + 1]):
        vort_c -= fy[0, 0, 0]

    return vort_c


@gtscript.function
def update_vorticity_and_kinetic_energy(
    ua: FloatField,
    va: FloatField,
    uc: FloatField,
    vc: FloatField,
    u: FloatField,
    v: FloatField,
    sin_sg1: FloatField,
    cos_sg1: FloatField,
    sin_sg2: FloatField,
    cos_sg2: FloatField,
    sin_sg3: FloatField,
    cos_sg3: FloatField,
    sin_sg4: FloatField,
    cos_sg4: FloatField,
    dt2: float,
):
    from __externals__ import i_end, i_start, j_end, j_start, namelist

    assert __INLINED(namelist.grid_type < 3)

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

    return ke, vort


@gtscript.function
def vorticitytransport(
    vort: FloatField,
    ke: FloatField,
    u: FloatField,
    v: FloatField,
    uc: FloatField,
    vc: FloatField,
    cosa_u: FloatField,
    sina_u: FloatField,
    cosa_v: FloatField,
    sina_v: FloatField,
    rdxc: FloatField,
    rdyc: FloatField,
    dt2: float,
):
    from __externals__ import (
        i_end,
        i_start,
        j_end,
        j_start,
        local_ie,
        local_is,
        local_je,
        local_js,
        namelist,
    )

    assert __INLINED(namelist.grid_type < 3)
    # additional assumption: not __INLINED(spec.grid.nested)

    tmp_flux_zonal = dt2 * (v - uc * cosa_u) / sina_u
    with horizontal(region[i_start, :], region[i_end + 1, :]):
        tmp_flux_zonal = dt2 * v

    with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
        flux = vort[0, 0, 0] if tmp_flux_zonal > 0.0 else vort[0, 1, 0]
        uc = uc + tmp_flux_zonal * flux + rdxc * (ke[-1, 0, 0] - ke)

    tmp_flux_merid = dt2 * (u - vc * cosa_v) / sina_v
    with horizontal(region[:, j_start], region[:, j_end + 1]):
        tmp_flux_merid = dt2 * u

    with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
        flux = vort[0, 0, 0] if tmp_flux_merid > 0.0 else vort[1, 0, 0]
        vc = vc - tmp_flux_merid * flux + rdyc * (ke[0, -1, 0] - ke)

    return uc, vc


def csw_stencil(
    delpc: FloatField,
    ptc: FloatField,
    delp: FloatField,
    pt: FloatField,
    divgd: FloatField,
    cosa_s: FloatField,
    cosa_u: FloatField,
    cosa_v: FloatField,
    sina_u: FloatField,
    sina_v: FloatField,
    dx: FloatField,
    dy: FloatField,
    dxa: FloatField,
    dya: FloatField,
    dxc: FloatField,
    dyc: FloatField,
    rdxc: FloatField,
    rdyc: FloatField,
    rsin2: FloatField,
    rsin_u: FloatField,
    rsin_v: FloatField,
    sin_sg1: FloatField,
    sin_sg2: FloatField,
    sin_sg3: FloatField,
    sin_sg4: FloatField,
    cos_sg1: FloatField,
    cos_sg2: FloatField,
    cos_sg3: FloatField,
    cos_sg4: FloatField,
    rarea: FloatField,
    rarea_c: FloatField,
    fC: FloatField,
    u: FloatField,
    ua: FloatField,
    uc: FloatField,
    ut: FloatField,
    v: FloatField,
    va: FloatField,
    vc: FloatField,
    vt: FloatField,
    w: FloatField,
    omga: FloatField,
    dt2: float,
):
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(...):
        uc, vc, ua, va, ut, vt = d2a2c_vect(
            cosa_s,
            cosa_u,
            cosa_v,
            dxa,
            dya,
            rsin2,
            rsin_u,
            rsin_v,
            sin_sg1,
            sin_sg2,
            sin_sg3,
            sin_sg4,
            u,
            ua,
            uc,
            ut,
            v,
            va,
            vc,
            vt,
        )

        divgd_t = divergence_corner(
            u,
            v,
            ua,
            va,
            dxc,
            dyc,
            sin_sg1,
            sin_sg2,
            sin_sg3,
            sin_sg4,
            cos_sg1,
            cos_sg2,
            cos_sg3,
            cos_sg4,
            rarea_c,
        )

        # Extra -- for validation
        # {
        with horizontal(region[local_is : local_ie + 2, local_js : local_je + 2]):
            divgd = divgd_t
        # }

        ut = dt2 * ut * dy * sin_sg3[-1, 0, 0] if ut > 0 else dt2 * ut * dy * sin_sg1
        vt = dt2 * vt * dx * sin_sg4[0, -1, 0] if vt > 0 else dt2 * vt * dx * sin_sg2

        delpc_t, ptc_t, omga_t = transportdelp(delp, pt, ut, vt, w, rarea)

        # Extra -- for validation
        # {
        with horizontal(
            region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
        ):
            delpc = delpc_t
            ptc = ptc_t
            omga = omga_t
        # }

        ke, vort = update_vorticity_and_kinetic_energy(
            ua,
            va,
            uc,
            vc,
            u,
            v,
            sin_sg1,
            cos_sg1,
            sin_sg2,
            cos_sg2,
            sin_sg3,
            cos_sg3,
            sin_sg4,
            cos_sg4,
            dt2,
        )

        vort = fC + rarea_c * circulation_cgrid(uc, vc, dxc, dyc)

        uc, vc = vorticitytransport(
            vort, ke, u, v, uc, vc, cosa_u, sina_u, cosa_v, sina_v, rdxc, rdyc, dt2
        )


def compute(delp, pt, u, v, w, uc, vc, ua, va, ut, vt, divgd, omga, dt2):
    grid = spec.grid
    dord4 = True
    origin_halo1 = (grid.is_ - 1, grid.js - 1, 0)
    delpc = utils.make_storage_from_shape(delp.shape, origin=origin_halo1)
    ptc = utils.make_storage_from_shape(pt.shape, origin=origin_halo1)

    if spec.namelist.npx != spec.namelist.npy:
        raise NotImplementedError("D2A2C assumes a square grid")
    if spec.namelist.npx <= 13 and spec.namelist.layout[0] > 1:
        D2A2C_AVG_OFFSET = -1
    else:
        D2A2C_AVG_OFFSET = 3

    stencil = gtstencil(
        definition=csw_stencil,
        externals={"D2A2C_AVG_OFFSET": D2A2C_AVG_OFFSET},
    )

    stencil(
        delpc,
        ptc,
        delp,
        pt,
        divgd,
        grid.cosa_s,
        grid.cosa_u,
        grid.cosa_v,
        grid.sina_u,
        grid.sina_v,
        grid.dx,
        grid.dy,
        grid.dxa,
        grid.dya,
        grid.dxc,
        grid.dyc,
        grid.rdxc,
        grid.rdyc,
        grid.rsin2,
        grid.rsin_u,
        grid.rsin_v,
        grid.sin_sg1,
        grid.sin_sg2,
        grid.sin_sg3,
        grid.sin_sg4,
        grid.cos_sg1,
        grid.cos_sg2,
        grid.cos_sg3,
        grid.cos_sg4,
        grid.rarea,
        grid.rarea_c,
        grid.fC,
        u,
        ua,
        uc,
        ut,
        v,
        va,
        vc,
        vt,
        w,
        omga,
        dt2,
        origin=grid.compute_origin(add=(-1, -1, 0)),
        domain=grid.domain_shape_compute(add=(3, 3, 0)),
    )

    return delpc, ptc
