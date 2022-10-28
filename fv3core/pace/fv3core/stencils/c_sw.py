from gt4py.gtscript import (
    PARALLEL,
    compile_assert,
    computation,
    horizontal,
    interval,
    region,
)

import pace.dsl.gt4py_utils as utils
from pace.dsl.dace.orchestration import orchestrate
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ
from pace.fv3core.stencils.basic_operations import compute_coriolis_parameter_defn
from pace.fv3core.stencils.d2a2c_vect import DGrid2AGrid2CGridVectors
from pace.stencils import corners
from pace.util import X_DIM, X_INTERFACE_DIM, Y_DIM, Y_INTERFACE_DIM, Z_DIM
from pace.util.grid import GridData


def zero_delpc_ptc(delpc: FloatField, ptc: FloatField):
    """
    Args:
        delpc (out):
        ptc (out):
    """
    with computation(PARALLEL), interval(...):
        delpc = 0.0
        ptc = 0.0


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

    Computed in order to damp the divergence, the damping is applied
    at the end of d_sw directly to the winds
    Described in detail in chapter 8 (?) of the FV3 docs

    Args:
        u (in): x-velocity on d-grid
        v (in): y-velocity on d-grid
        ua (in): x-velocity on a
        va (in): y-velocity on a
        dxc (in): grid spacing in x-direction
        dyc (in): grid spacing in y-direction
        sin_sg1 (in): grid sin(sg1)
        sin_sg2 (in): grid sin(sg2)
        sin_sg3 (in): grid sin(sg3)
        sin_sg4 (in): grid sin(sg4)
        cos_sg1 (in): grid cos(sg1)
        cos_sg2 (in): grid cos(sg2)
        cos_sg3 (in): grid cos(sg3)
        cos_sg4 (in): grid cos(sg4)
        rarea_c (in): inverse cell areas on c-grid
        divg_d (out): divergence on d-grid (cell corners)
    """
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        uf = (
            (u - 0.25 * (va[0, -1, 0] + va) * (cos_sg4[0, -1] + cos_sg2))
            * dyc
            * 0.5
            * (sin_sg4[0, -1] + sin_sg2)
        )
        """c-grid (?) contravariant component of the wind in the x-direction"""
        # TODO: refactor this into a call to contravariant()
        # TODO: add reference to FV3 documentation on divergence damping

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
            vf0 = v * dxc * 0.5 * (sin_sg3[-1, 0] + sin_sg1)
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
            divg_d = (vf1 - vf0 + uf1 - uf) * rarea_c

        with horizontal(region[:, j_start], region[:, j_end + 1]):
            uf0 = u * dyc * 0.5 * (sin_sg4[0, -1] + sin_sg2)
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
            divg_d = (vf1 - vf + uf1 - uf0) * rarea_c

        with horizontal(region[i_start, j_start], region[i_end + 1, j_start]):
            uf1 = u[-1, 0, 0] * dyc[-1, 0] * 0.5 * (sin_sg4[-1, -1] + sin_sg2[-1, 0])
            vf0 = v * dxc * 0.5 * (sin_sg3[-1, 0] + sin_sg1)
            uf0 = u * dyc * 0.5 * (sin_sg4[0, -1] + sin_sg2)
            divg_d = (-vf0 + uf1 - uf0) * rarea_c

        with horizontal(region[i_end + 1, j_end + 1], region[i_start, j_end + 1]):
            vf1 = v[0, -1, 0] * dxc[0, -1] * 0.5 * (sin_sg3[-1, -1] + sin_sg1[0, -1])
            uf1 = u[-1, 0, 0] * dyc[-1, 0] * 0.5 * (sin_sg4[-1, -1] + sin_sg2[-1, 0])
            uf0 = u * dyc * 0.5 * (sin_sg4[0, -1] + sin_sg2)
            divg_d = (vf1 + uf1 - uf0) * rarea_c

        # ---------


def geoadjust_ut(
    ut: FloatField,
    dy: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    sin_sg1: FloatFieldIJ,
    dt2: float,
):
    """
    take c-grid contravariant wind to compute the 1st order upwind flux
    Args:
        ut (out): c-grid contravariant u * dx as input, as output
            includes metric terms (dt2*dy*sin_sg) needed to compute fluxes,
            "volume flux" (?)
        dy (in):
        sin_sg3 (in):
        sin_sg1 (in):
        dt2 (in):
    """
    # TODO: describe what ut (out) is as a word or description, and propagate
    # it to the rest of the code using it
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
    """
    Args:
        vt (out):
        dx (in):
        sin_sg4 (in):
        sin_sg2 (in):
        dt2 (in):
    """
    with computation(PARALLEL), interval(...):
        vt[0, 0, 0] = (
            dt2 * vt * dx * sin_sg4[0, -1] if vt > 0 else dt2 * vt * dx * sin_sg2
        )


def fill_corners_delp_pt_w(
    delp_in: FloatField,
    pt_in: FloatField,
    w_in: FloatField,
    delp_out: FloatField,
    pt_out: FloatField,
    w_out: FloatField,
):
    """
    Args:
        delp_in (in):
        pt_in (in):
        w_in (in):
        delp_out (out):
        pt_out (out):
        w_out (out):
    """
    from __externals__ import fill_corners_func

    with computation(PARALLEL), interval(...):
        delp_out = fill_corners_func(delp_in)
        pt_out = fill_corners_func(pt_in)
        w_out = fill_corners_func(w_in)


def compute_nonhydrostatic_fluxes_x(
    delp: FloatField,
    pt: FloatField,
    utc: FloatField,
    w: FloatField,
    fx: FloatField,
    fx1: FloatField,
    fx2: FloatField,
):
    """
    Computes scalar fluxes
    Args:
        delp (in):
        pt (in):
        utc (in): metric and wind speed term of the first-order upwind flux
        w (in):
        fx (out): heat (entropy) flux, first-order upwind flux of delp
            in units per second
        fx1 (out): first-order upwind flux of delp in units per second
        fx2 (out): flux of veritcal momentum, first-order upwind flux of w
            in units per second
    """
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

    steps delpc, ptc, wc.
    Diagnostically computes ke, vorticity for the start of the timestep

    Args:
        delp (in): what is transported
        pt (in): pressure
        utc (???): x-velocity on C-grid
        vtc (in): y-velocity on C-grid
        w (in): z-velocity on C-grid
        rarea (in): inverse gridcell area
        delpc (out): updated delp
        ptc (out): updated pt
        wc (out): updated w
        ke (out): kinetic energy, computed with an upstream bias to avoid an instability
        vort (out): vorticity
        ua (in): u wind on A-grid
        va (in): v wind on A-grid
        uc (in): u wind on C-grid
        vc (in): v wind on C-grid
        u (in): u wind on D-grid
        v (in): v wind on D-grid
        fx (in):
        fx1 (in):
        fx2 (in):
        sin_sg1 (in):
        cos_sg1 (in):
        sin_sg2 (in):
        cos_sg2 (in):
        sin_sg3 (in):
        cos_sg3 (in):
        sin_sg4 (in):
        cos_sg4 (in):
        dt2 (in): length of half a timestep
    """

    from __externals__ import grid_type, i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        compile_assert(grid_type < 3)
        # additional assumption (not grid.nested)
        # corresponds to x fluxes function, but for y-direction
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


def circulation_cgrid(
    uc: FloatField,
    vc: FloatField,
    dxc: FloatFieldIJ,
    dyc: FloatFieldIJ,
    vort_c: FloatField,
):
    """Diagnostically compute vort_c.

    Args:
        uc (in): x-velocity on C-grid
        vc (in): y-velocity on C-grid
        dxc (in): grid spacing in x-dir
        dyc (in): grid spacing in y-dir
        vort_c (out): C-grid relative vorticity
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


def absolute_vorticity(vort: FloatField, fC: FloatFieldIJ, rarea_c: FloatFieldIJ):
    """
    Args:
        vort (out): absolute vorticity
        fC (in):
        rarea_c (in):
    """
    with computation(PARALLEL), interval(...):
        vort[0, 0, 0] = fC + rarea_c * vort


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
    """
    Args:
        vorticity (in):
        ke (in):
        velocity (in):
        velocity_c (out):
        cosa (in):
        sina (in):
        rdxc (in):
    """
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
    """
    Args:
        vorticity (in): absolute vorticity
        ke (in):
        velocity (in):
        velocity_c (out): velocity update in y-direction
        cosa (in):
        sina (in):
        rdyc (in):
    """
    from __externals__ import grid_type, j_end, j_start

    with computation(PARALLEL), interval(...):
        compile_assert(grid_type < 3)
        # additional assumption: not __INLINED(spec.grid.nested)

        # first-order upwind voriticity flux
        tmp_flux = dt2 * (velocity - velocity_c * cosa) / sina
        with horizontal(region[:, j_start], region[:, j_end + 1]):
            tmp_flux = dt2 * velocity

        flux = vorticity[0, 0, 0] if tmp_flux > 0.0 else vorticity[1, 0, 0]
        # forward-stepped y velocity
        # time derivative of vector horizontal wind is equal to curl of vorticity
        # try to line up with Chapter 5 & 6 of the FV3 documentation
        velocity_c = velocity_c - tmp_flux * flux + rdyc * (ke[0, -1, 0] - ke)


def compute_fC(
    stencil_factory: StencilFactory, lon: FloatFieldIJ, lat: FloatFieldIJ, backend: str
):
    """
    Compute the coriolis parameter on the C-grid
    """
    fC = utils.make_storage_from_shape(lon.shape, backend=backend)
    fC_stencil = stencil_factory.from_dims_halo(
        compute_coriolis_parameter_defn,
        compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
        compute_halos=(3, 3),
    )
    fC_stencil(fC, lon, lat, 0.0)
    return fC


class CGridShallowWaterDynamics:
    """
    Fortran name is c_sw
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        grid_data: GridData,
        nested: bool,
        grid_type: int,
        nord: int,
    ):
        orchestrate(obj=self, config=stencil_factory.config.dace_config)
        grid_indexing = stencil_factory.grid_indexing
        self.grid_data = grid_data
        self._dord4 = True
        self._fC = compute_fC(
            stencil_factory,
            self.grid_data.lon,
            self.grid_data.lat,
            backend=stencil_factory.backend,
        )
        origin_halo1 = (grid_indexing.isc - 1, grid_indexing.jsc - 1, 0)
        self.delpc = utils.make_storage_from_shape(
            grid_indexing.max_shape,
            origin=origin_halo1,
            backend=stencil_factory.backend,
        )
        """
        pressure thickness on c-grid forward step
        """
        self.ptc = utils.make_storage_from_shape(
            grid_indexing.max_shape,
            origin=origin_halo1,
            backend=stencil_factory.backend,
        )
        """
        potential temperature on c-grid forward step
        """
        self._zero_delpc_ptc = stencil_factory.from_dims_halo(
            zero_delpc_ptc,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
            compute_halos=(3, 3),
        )

        self._D2A2CGrid_Vectors = DGrid2AGrid2CGridVectors(
            stencil_factory,
            grid_data,
            nested,
            grid_type,
            self._dord4,
        )

        def make_storage():
            return utils.make_storage_from_shape(
                grid_indexing.max_shape,
                backend=stencil_factory.backend,
            )

        self._tmp_ke = make_storage()
        self._tmp_vort = make_storage()
        self._tmp_fx = make_storage()
        self._tmp_fx1 = make_storage()
        self._tmp_fx2 = make_storage()

        if nord > 0:
            self._divergence_corner = stencil_factory.from_dims_halo(
                func=divergence_corner,
                compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
            )
        else:
            self._divergence_corner = None

        self._geoadjust_ut = stencil_factory.from_dims_halo(
            func=geoadjust_ut,
            compute_dims=[X_INTERFACE_DIM, Y_DIM, Z_DIM],
            compute_halos=(1, 1),
        )

        self._geoadjust_vt = stencil_factory.from_dims_halo(
            func=geoadjust_vt,
            compute_dims=[X_DIM, Y_INTERFACE_DIM, Z_DIM],
            compute_halos=(1, 1),
        )

        self._fill_corners_x_delp_pt_w_stencil = stencil_factory.from_dims_halo(
            fill_corners_delp_pt_w,
            externals={"fill_corners_func": corners.fill_corners_2cells_x},
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
            compute_halos=(3, 3),
        )

        self._compute_nonhydro_fluxes_x_stencil = stencil_factory.from_dims_halo(
            compute_nonhydrostatic_fluxes_x,
            compute_dims=[X_INTERFACE_DIM, Y_DIM, Z_DIM],
            compute_halos=(1, 1),
        )

        self._fill_corners_y_delp_pt_w_stencil = stencil_factory.from_dims_halo(
            fill_corners_delp_pt_w,
            externals={"fill_corners_func": corners.fill_corners_2cells_y},
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
            compute_halos=(3, 3),
        )

        self._transportdelp_updatevorticity_and_ke = stencil_factory.from_dims_halo(
            func=transportdelp_update_vorticity_and_kineticenergy,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
            compute_halos=(1, 1),
            externals={"grid_type": grid_type},
        )

        self._circulation_cgrid = stencil_factory.from_dims_halo(
            func=circulation_cgrid,
            compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
        )
        self._absolute_vorticity = stencil_factory.from_dims_halo(
            func=absolute_vorticity,
            compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
        )

        self._update_y_velocity = stencil_factory.from_dims_halo(
            func=update_y_velocity,
            compute_dims=[X_DIM, Y_INTERFACE_DIM, Z_DIM],
            externals={
                "grid_type": grid_type,
            },
        )

        self._update_x_velocity = stencil_factory.from_dims_halo(
            func=update_x_velocity,
            compute_dims=[X_INTERFACE_DIM, Y_DIM, Z_DIM],
            externals={"grid_type": grid_type},
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

        Also advances delp and pt half a timestep, get computed
        in order to compute the pressure gradient force, since we need that
        on the half-step for D_SW.

        Args:
            delp (in): D-grid vertical delta in pressure
            pt (inout): D-grid potential temperature (only halos get updated)
            u (in): D-grid x-velocity
            v (in): D-grid y-velocity
            w (in): vertical velocity
            uc (inout): C-grid x-velocity
            vc (inout): C-grid y-velocity
            ua (inout): A-grid x-velocity
            va (inout): A-grid y-velocity
            ut (out): u * dx
            vt (out): v * dy
            divgd (out): D-grid horizontal divergence
                Computed in order to damp the divergence, the damping is applied
                at the end of d_sw directly to the winds
            omga (out): Vertical pressure velocity
            dt2 (in): Half a model timestep in seconds
        """
        # TODO: omga is called "wc" inside stencils, consolidate the naming
        self._zero_delpc_ptc(
            self.delpc,
            self.ptc,
        )
        self._D2A2CGrid_Vectors(uc, vc, u, v, ua, va, ut, vt)
        if self._divergence_corner is not None:
            self._divergence_corner(
                u,
                v,
                ua,
                va,
                self.grid_data.dxc,
                self.grid_data.dyc,
                self.grid_data.sin_sg1,
                self.grid_data.sin_sg2,
                self.grid_data.sin_sg3,
                self.grid_data.sin_sg4,
                self.grid_data.cos_sg1,
                self.grid_data.cos_sg2,
                self.grid_data.cos_sg3,
                self.grid_data.cos_sg4,
                self.grid_data.rarea_c,
                divgd,
            )

        # TODO: what does "geoadjust" mean?
        # TODO: merge geoadjust routines into compute_nonhydro_fluxes, if possible
        self._geoadjust_ut(
            ut,
            self.grid_data.dy,
            self.grid_data.sin_sg3,
            self.grid_data.sin_sg1,
            dt2,
        )
        self._geoadjust_vt(
            vt,
            self.grid_data.dx,
            self.grid_data.sin_sg4,
            self.grid_data.sin_sg2,
            dt2,
        )

        # TODO(eddied): We pass the same fields 2x to avoid GTC validation errors
        self._fill_corners_x_delp_pt_w_stencil(delp, pt, w, delp, pt, w)
        # TODO: why is there only a "x" version of this? Is the "y" verison folded
        # into the next routine?
        self._compute_nonhydro_fluxes_x_stencil(
            delp, pt, ut, w, self._tmp_fx, self._tmp_fx1, self._tmp_fx2
        )
        self._fill_corners_y_delp_pt_w_stencil(delp, pt, w, delp, pt, w)
        self._transportdelp_updatevorticity_and_ke(
            delp,
            pt,
            ut,
            vt,
            w,
            self.grid_data.rarea,
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
            self.grid_data.sin_sg1,
            self.grid_data.cos_sg1,
            self.grid_data.sin_sg2,
            self.grid_data.cos_sg2,
            self.grid_data.sin_sg3,
            self.grid_data.cos_sg3,
            self.grid_data.sin_sg4,
            self.grid_data.cos_sg4,
            dt2,
        )
        # TODO: we can merge circulation cgrid and absolute vorticty
        # into a single stencil
        self._circulation_cgrid(
            uc,
            vc,
            self.grid_data.dxc,
            self.grid_data.dyc,
            self._tmp_vort,
        )
        self._absolute_vorticity(
            self._tmp_vort,
            self._fC,
            self.grid_data.rarea_c,
        )
        self._update_y_velocity(
            self._tmp_vort,
            self._tmp_ke,
            u,
            vc,
            self.grid_data.cosa_v,
            self.grid_data.sina_v,
            self.grid_data.rdyc,
            dt2,
        )
        self._update_x_velocity(
            self._tmp_vort,
            self._tmp_ke,
            v,
            uc,
            self.grid_data.cosa_u,
            self.grid_data.sina_u,
            self.grid_data.rdxc,
            dt2,
        )
        return self.delpc, self.ptc
