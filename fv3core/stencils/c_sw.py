from gt4py.gtscript import (
    PARALLEL,
    compile_assert,
    computation,
    horizontal,
    interval,
    region,
)

import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil
from fv3core.stencils.d2a2c_vect import DGrid2AGrid2CGridVectors
from fv3core.utils import corners
from fv3core.utils.grid import axis_offsets
from fv3core.utils.typing import FloatField, FloatFieldIJ


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


def absolute_vorticity(vort: FloatField, fC: FloatFieldIJ, rarea_c: FloatFieldIJ):
    with computation(PARALLEL), interval(...):
        vort[0, 0, 0] = fC + rarea_c * vort


def fill_corners_delp_pt_w(
    delp: FloatField,
    pt: FloatField,
    w: FloatField,
):
    from __externals__ import fill_corners_func

    with computation(PARALLEL), interval(...):
        delp = fill_corners_func(delp)
        pt = fill_corners_func(pt)
        w = fill_corners_func(w)


def compute_nonhydro_fluxes_x(
    delp: FloatField,
    pt: FloatField,
    utc: FloatField,
    w: FloatField,
    fx: FloatField,
    fx1: FloatField,
    fx2: FloatField,
):
    with computation(PARALLEL), interval(...):
        fx1 = delp[-1, 0, 0] if utc > 0.0 else delp
        fx = pt[-1, 0, 0] if utc > 0.0 else pt
        fx2 = w[-1, 0, 0] if utc > 0.0 else w
        fx1 = utc * fx1
        fx = fx1 * fx
        fx2 = fx1 * fx2


def transportdelp_update_vorticity_and_kineticenergy(
    delp: FloatField,
    pt: FloatField,
    utc: FloatField,
    vtc: FloatField,
    w: FloatField,
    rarea: FloatFieldIJ,
    delpc: FloatField,
    ptc: FloatField,
    wc: FloatField,
    ke: FloatField,
    vort: FloatField,
    ua: FloatField,
    va: FloatField,
    uc: FloatField,
    vc: FloatField,
    u: FloatField,
    v: FloatField,
    fx: FloatField,
    fx1: FloatField,
    fx2: FloatField,
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
    """Transport delp then update vorticity and kinetic energy

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
        ke: kinetic energy (inout)
        vort: vorticity (inout)
        ua/uc/u: u wind on the a/c/d grid (in)
        va/vc/v: v wind on the a/c/d grid (in)
        sin_sg/cos_sg 1/2/3/4: variables that specify grid geometry grid (in)
        dt2: length of half a timestep (in)
    """

    from __externals__ import grid_type, i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        compile_assert(grid_type < 3)
        # additional assumption (not grid.nested)
        fy1 = delp[0, -1, 0] if vtc > 0.0 else delp
        fy = pt[0, -1, 0] if vtc > 0.0 else pt
        fy2 = w[0, -1, 0] if vtc > 0.0 else w
        fy1 = vtc * fy1
        fy = fy1 * fy
        fy2 = fy1 * fy2

        delpc = delp + (fx1 - fx1[1, 0, 0] + fy1 - fy1[0, 1, 0]) * rarea
        ptc = (pt * delp + (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea) / delpc
        wc = (w * delp + (fx2 - fx2[1, 0, 0] + fy2 - fy2[0, 1, 0]) * rarea) / delpc

    with computation(PARALLEL), interval(...):
        # update vorticity and kinetic energy
        compile_assert(grid_type < 3)

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

        vf = (
            (v - 0.25 * (ua[-1, 0, 0] + ua) * (cos_sg3[-1, 0] + cos_sg1))
            * dxc
            * 0.5
            * (sin_sg3[-1, 0] + sin_sg1)
        )

        divg_d = (vf[0, -1, 0] - vf + uf[-1, 0, 0] - uf) * rarea_c

        # The original code is:
        # ---------
        # with horizontal(region[:, j_start], region[:, j_end + 1]):
        #     uf = u * dyc * 0.5 * (sin_sg4[0, -1] + sin_sg2)
        # with horizontal(region[i_start, :], region[i_end + 1, :]):
        #     vf = v * dxc * 0.5 * (sin_sg3[-1, 0] + sin_sg1)
        # with horizontal(region[i_start, j_start], region[i_end + 1, j_start]):
        #     divg_d = (-vf + uf[-1, 0, 0] - uf) * rarea_c
        # with horizontal(region[i_end + 1, j_end + 1], region[i_start, j_end + 1]):
        #     divg_d = (vf[0, -1, 0] + uf[-1, 0, 0] - uf) * rarea_c
        # ---------
        #
        # Code with regions restrictions:
        # ---------
        # variables ending with 1 are the shifted versions
        # in the future we could use gtscript functions when they support shifts

        with horizontal(region[i_start, :], region[i_end + 1, :]):
            vf = v * dxc * 0.5 * (sin_sg3[-1, 0] + sin_sg1)
            vf1 = v[0, -1, 0] * dxc[0, -1] * 0.5 * (sin_sg3[-1, -1] + sin_sg1[0, -1])
            uf1 = (
                (
                    u[-1, 0, 0]
                    - 0.25
                    * (va[-1, -1, 0] + va[-1, 0, 0])
                    * (cos_sg4[-1, -1] + cos_sg2[-1, 0])
                )
                * dyc[-1, 0]
                * 0.5
                * (sin_sg4[-1, -1] + sin_sg2[-1, 0])
            )
            divg_d = (vf1 - vf + uf1 - uf) * rarea_c

        with horizontal(region[:, j_start], region[:, j_end + 1]):
            uf = u * dyc * 0.5 * (sin_sg4[0, -1] + sin_sg2)
            uf1 = u[-1, 0, 0] * dyc[-1, 0] * 0.5 * (sin_sg4[-1, -1] + sin_sg2[-1, 0])
            vf1 = (
                (
                    v[0, -1, 0]
                    - 0.25
                    * (ua[-1, -1, 0] + ua[0, -1, 0])
                    * (cos_sg3[-1, -1] + cos_sg1[0, -1])
                )
                * dxc[0, -1]
                * 0.5
                * (sin_sg3[-1, -1] + sin_sg1[0, -1])
            )
            divg_d = (vf1 - vf + uf1 - uf) * rarea_c

        with horizontal(region[i_start, j_start], region[i_end + 1, j_start]):
            vf = v * dxc * 0.5 * (sin_sg3[-1, 0] + sin_sg1)
            uf = u * dyc * 0.5 * (sin_sg4[0, -1] + sin_sg2)
            uf1 = u[-1, 0, 0] * dyc[-1, 0] * 0.5 * (sin_sg4[-1, -1] + sin_sg2[-1, 0])
            divg_d = (-vf + uf1 - uf) * rarea_c

        with horizontal(region[i_end + 1, j_end + 1], region[i_start, j_end + 1]):
            vf1 = v[0, -1, 0] * dxc[0, -1] * 0.5 * (sin_sg3[-1, -1] + sin_sg1[0, -1])
            uf1 = u[-1, 0, 0] * dyc[-1, 0] * 0.5 * (sin_sg4[-1, -1] + sin_sg2[-1, 0])
            uf = u * dyc * 0.5 * (sin_sg4[0, -1] + sin_sg2)
            divg_d = (vf1 + uf1 - uf) * rarea_c

        # ---------


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
        # fx1 and fy1 are the shifted versions of fx and fy and are defined
        # because temporaries are not allowed to be accessed with offsets in regions.
        fx1 = dxc[0, -1] * uc[0, -1, 0]
        fy1 = dyc[-1, 0] * vc[-1, 0, 0]

        vort_c = fx1 - fx - fy1 + fy
        with horizontal(region[i_start, j_start], region[i_start, j_end + 1]):
            vort_c = fx1 - fx + fy
        with horizontal(region[i_end + 1, j_start], region[i_end + 1, j_end + 1]):
            vort_c = fx1 - fx - fy1


def update_x_velocity(
    vorticity: FloatField,
    ke: FloatField,
    velocity: FloatField,
    velocity_c: FloatField,
    cosa: FloatFieldIJ,
    sina: FloatFieldIJ,
    rdxc: FloatFieldIJ,
    dt2: float,
):
    from __externals__ import grid_type, i_end, i_start

    with computation(PARALLEL), interval(...):
        compile_assert(grid_type < 3)
        # additional assumption: not __INLINED(spec.grid.nested)

        tmp_flux = dt2 * (velocity - velocity_c * cosa) / sina
        with horizontal(region[i_start, :], region[i_end + 1, :]):
            tmp_flux = dt2 * velocity

        flux = vorticity[0, 0, 0] if tmp_flux > 0.0 else vorticity[0, 1, 0]
        velocity_c = velocity_c + tmp_flux * flux + rdxc * (ke[-1, 0, 0] - ke)


def update_y_velocity(
    vorticity: FloatField,
    ke: FloatField,
    velocity: FloatField,
    velocity_c: FloatField,
    cosa: FloatFieldIJ,
    sina: FloatFieldIJ,
    rdyc: FloatFieldIJ,
    dt2: float,
):
    from __externals__ import grid_type, j_end, j_start

    with computation(PARALLEL), interval(...):
        compile_assert(grid_type < 3)
        # additional assumption: not __INLINED(spec.grid.nested)

        tmp_flux = dt2 * (velocity - velocity_c * cosa) / sina
        with horizontal(region[:, j_start], region[:, j_end + 1]):
            tmp_flux = dt2 * velocity

        flux = vorticity[0, 0, 0] if tmp_flux > 0.0 else vorticity[1, 0, 0]
        velocity_c = velocity_c - tmp_flux * flux + rdyc * (ke[0, -1, 0] - ke)


def initialize_delpc_ptc(delpc: FloatField, ptc: FloatField):
    with computation(PARALLEL), interval(...):
        delpc = 0.0
        ptc = 0.0


class CGridShallowWaterDynamics:
    """
    Fortran name is c_sw
    """

    def __init__(self, grid, namelist):
        self.grid = grid
        self.namelist = namelist
        self._dord4 = True

        self._D2A2CGrid_Vectors = DGrid2AGrid2CGridVectors(
            self.grid.grid_indexing,
            self.grid.cosa_s,
            self.grid.cosa_u,
            self.grid.cosa_v,
            self.grid.rsin_u,
            self.grid.rsin_v,
            self.grid.rsin2,
            self.grid.dxa,
            self.grid.dya,
            self.grid.sin_sg1,
            self.grid.sin_sg2,
            self.grid.sin_sg3,
            self.grid.sin_sg4,
            self.grid.nested,
            self.namelist.grid_type,
            self._dord4,
        )
        grid_type = self.namelist.grid_type
        origin_halo1 = (self.grid.is_ - 1, self.grid.js - 1, 0)
        shape = self.grid.domain_shape_full(add=(1, 1, 1))
        self.delpc = utils.make_storage_from_shape(shape, origin=origin_halo1)
        self.ptc = utils.make_storage_from_shape(shape, origin=origin_halo1)
        self._initialize_delpc_ptc = FrozenStencil(
            func=initialize_delpc_ptc,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(),
        )

        self._tmp_ke = utils.make_storage_from_shape(shape)
        self._tmp_vort = utils.make_storage_from_shape(shape)
        self._tmp_fx = utils.make_storage_from_shape(shape)
        self._tmp_fx1 = utils.make_storage_from_shape(shape)
        self._tmp_fx2 = utils.make_storage_from_shape(shape)
        origin = self.grid.compute_origin()
        domain = self.grid.domain_shape_compute(add=(1, 1, 0))
        ax_offsets = axis_offsets(self.grid, origin, domain)

        if self.namelist.nord > 0:
            self._divergence_corner = FrozenStencil(
                func=divergence_corner,
                externals={
                    **ax_offsets,
                },
                origin=origin,
                domain=domain,
            )
        geo_origin = (self.grid.is_ - 1, self.grid.js - 1, 0)
        self._geoadjust_ut = FrozenStencil(
            func=geoadjust_ut,
            origin=geo_origin,
            domain=(self.grid.nic + 3, self.grid.njc + 2, self.grid.npz),
        )
        self._geoadjust_vt = FrozenStencil(
            func=geoadjust_vt,
            origin=geo_origin,
            domain=(self.grid.nic + 2, self.grid.njc + 3, self.grid.npz),
        )
        origin_full = self.grid.full_origin()
        domain_full = self.grid.domain_shape_full()
        ax_offsets_full = axis_offsets(self.grid, origin_full, domain_full)

        self._fill_corners_x_delp_pt_w_stencil = FrozenStencil(
            fill_corners_delp_pt_w,
            externals={
                "fill_corners_func": corners.fill_corners_2cells_x,
                **ax_offsets_full,
            },
            origin=origin_full,
            domain=domain_full,
        )
        self._fill_corners_y_delp_pt_w_stencil = FrozenStencil(
            fill_corners_delp_pt_w,
            externals={
                "fill_corners_func": corners.fill_corners_2cells_y,
                **ax_offsets_full,
            },
            origin=origin_full,
            domain=domain_full,
        )
        self._compute_nonhydro_fluxes_x_stencil = FrozenStencil(
            compute_nonhydro_fluxes_x,
            origin=geo_origin,
            domain=(self.grid.nic + 3, self.grid.njc + 2, self.grid.npz),
        )

        domain_transportdelp = (self.grid.nic + 2, self.grid.njc + 2, self.grid.npz)
        ax_offsets_transportdelp = axis_offsets(
            self.grid, geo_origin, domain_transportdelp
        )

        self._transportdelp_updatevorticity_and_ke = FrozenStencil(
            func=transportdelp_update_vorticity_and_kineticenergy,
            externals={
                "grid_type": grid_type,
                **ax_offsets_transportdelp,
            },
            origin=geo_origin,
            domain=domain_transportdelp,
        )
        self._circulation_cgrid = FrozenStencil(
            func=circulation_cgrid,
            externals={
                **ax_offsets,
            },
            origin=origin,
            domain=domain,
        )
        self._absolute_vorticity = FrozenStencil(
            func=absolute_vorticity,
            origin=origin,
            domain=(self.grid.nic + 1, self.grid.njc + 1, self.grid.npz),
        )

        domain_y = self.grid.domain_shape_compute(add=(0, 1, 0))
        axis_offsets_y = axis_offsets(self.grid, origin, domain_y)
        self._update_y_velocity = FrozenStencil(
            func=update_y_velocity,
            externals={
                "grid_type": grid_type,
                "j_start": axis_offsets_y["j_start"],
                "j_end": axis_offsets_y["j_end"],
            },
            origin=origin,
            domain=domain_y,
        )
        domain_x = self.grid.domain_shape_compute(add=(1, 0, 0))
        axis_offsets_x = axis_offsets(self.grid, origin, domain_x)
        self._update_x_velocity = FrozenStencil(
            func=update_x_velocity,
            externals={
                "grid_type": grid_type,
                "i_start": axis_offsets_x["i_start"],
                "i_end": axis_offsets_x["i_end"],
            },
            origin=origin,
            domain=domain_x,
        )

    def _vorticitytransport_cgrid(
        self,
        uc: FloatField,
        vc: FloatField,
        vort_c: FloatField,
        ke_c: FloatField,
        v: FloatField,
        u: FloatField,
        dt2: float,
    ):
        """Update the C-Grid x and y velocity fields.

        Args:
            uc: x-velocity on C-grid (input, output)
            vc: y-velocity on C-grid (input, output)
            vort_c: Vorticity on C-grid (input)
            ke_c: kinetic energy on C-grid (input)
            v: y-velocity on D-grid (input)
            u: x-velocity on D-grid (input)
            dt2: timestep (input)
        """
        self._update_y_velocity(
            vort_c,
            ke_c,
            u,
            vc,
            self.grid.cosa_v,
            self.grid.sina_v,
            self.grid.rdyc,
            dt2,
        )
        self._update_x_velocity(
            vort_c,
            ke_c,
            v,
            uc,
            self.grid.cosa_u,
            self.grid.sina_u,
            self.grid.rdxc,
            dt2,
        )

    def __call__(
        self,
        delp: FloatField,
        pt: FloatField,
        u: FloatField,
        v: FloatField,
        w: FloatField,
        uc: FloatField,
        vc: FloatField,
        ua: FloatField,
        va: FloatField,
        ut: FloatField,
        vt: FloatField,
        divgd: FloatField,
        omga: FloatField,
        dt2: float,
    ):
        """
        C-grid shallow water routine.

        Advances C-grid winds by half a time step.
        Args:
            delp: D-grid vertical delta in pressure (in)
            pt: D-grid potential temperature (in)
            u: D-grid x-velocity (in)
            v: D-grid y-velocity (in)
            w: vertical velocity (in)
            uc: C-grid x-velocity (inout)
            vc: C-grid y-velocity (inout)
            ua: A-grid x-velocity (in)
            va: A-grid y-velocity (in)
            ut: u * dx (inout)
            vt: v * dy (inout)
            divgd: D-grid horizontal divergence (inout)
            omga: Vertical pressure velocity (inout)
            dt2: Half a model timestep in seconds (in)
        """
        self._initialize_delpc_ptc(
            self.delpc,
            self.ptc,
        )
        self._D2A2CGrid_Vectors(uc, vc, u, v, ua, va, ut, vt)
        if self.namelist.nord > 0:
            self._divergence_corner(
                u,
                v,
                ua,
                va,
                self.grid.dxc,
                self.grid.dyc,
                self.grid.sin_sg1,
                self.grid.sin_sg2,
                self.grid.sin_sg3,
                self.grid.sin_sg4,
                self.grid.cos_sg1,
                self.grid.cos_sg2,
                self.grid.cos_sg3,
                self.grid.cos_sg4,
                self.grid.rarea_c,
                divgd,
            )
        self._geoadjust_ut(
            ut,
            self.grid.dy,
            self.grid.sin_sg3,
            self.grid.sin_sg1,
            dt2,
        )
        self._geoadjust_vt(
            vt,
            self.grid.dx,
            self.grid.sin_sg4,
            self.grid.sin_sg2,
            dt2,
        )

        self._fill_corners_x_delp_pt_w_stencil(delp, pt, w)
        self._compute_nonhydro_fluxes_x_stencil(
            delp, pt, ut, w, self._tmp_fx, self._tmp_fx1, self._tmp_fx2
        )
        self._fill_corners_y_delp_pt_w_stencil(delp, pt, w)
        self._transportdelp_updatevorticity_and_ke(
            delp,
            pt,
            ut,
            vt,
            w,
            self.grid.rarea,
            self.delpc,
            self.ptc,
            omga,
            self._tmp_ke,
            self._tmp_vort,
            ua,
            va,
            uc,
            vc,
            u,
            v,
            self._tmp_fx,
            self._tmp_fx1,
            self._tmp_fx2,
            self.grid.sin_sg1,
            self.grid.cos_sg1,
            self.grid.sin_sg2,
            self.grid.cos_sg2,
            self.grid.sin_sg3,
            self.grid.cos_sg3,
            self.grid.sin_sg4,
            self.grid.cos_sg4,
            dt2,
        )
        self._circulation_cgrid(
            uc,
            vc,
            self.grid.dxc,
            self.grid.dyc,
            self._tmp_vort,
        )
        self._absolute_vorticity(
            self._tmp_vort,
            self.grid.fC,
            self.grid.rarea_c,
        )
        self._vorticitytransport_cgrid(uc, vc, self._tmp_vort, self._tmp_ke, v, u, dt2)
        return self.delpc, self.ptc
