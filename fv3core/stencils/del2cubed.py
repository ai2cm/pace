from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

import fv3core._config as spec
import fv3core.utils.corners as corners
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil, get_stencils_with_varied_bounds
from fv3core.stencils.basic_operations import copy_defn
from fv3core.utils.grid import DampingCoefficients, GridIndexing, axis_offsets
from fv3core.utils.typing import FloatField, FloatFieldIJ
from fv3gfs.util import X_DIM, X_INTERFACE_DIM, Y_DIM, Y_INTERFACE_DIM, Z_DIM


#
# Flux value stencils
# ---------------------
def compute_zonal_flux(flux: FloatField, a_in: FloatField, del_term: FloatFieldIJ):
    with computation(PARALLEL), interval(...):
        flux = del_term * (a_in[-1, 0, 0] - a_in)


def compute_meridional_flux(flux: FloatField, a_in: FloatField, del_term: FloatFieldIJ):
    with computation(PARALLEL), interval(...):
        flux = del_term * (a_in[0, -1, 0] - a_in)


#
# Q update stencil
# ------------------
def update_q(
    q: FloatField, rarea: FloatFieldIJ, fx: FloatField, fy: FloatField, cd: float
):
    with computation(PARALLEL), interval(...):
        q += cd * rarea * (fx - fx[1, 0, 0] + fy - fy[0, 1, 0])


#
# corner_fill
#
# Stencil that copies/fills in the appropriate corner values for qdel
# ------------------------------------------------------------------------
def corner_fill(q_in: FloatField, q_out: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start

    # Fills the same scalar value into three locations in q for each corner
    with computation(PARALLEL), interval(...):
        third = 1.0 / 3.0

        q_out = q_in
        with horizontal(region[i_start, j_start]):
            q_out = (q_in[0, 0, 0] + q_in[-1, 0, 0] + q_in[0, -1, 0]) * third
        with horizontal(region[i_start - 1, j_start]):
            q_out = (q_in[1, 0, 0] + q_in[0, 0, 0] + q_in[1, -1, 0]) * third
        with horizontal(region[i_start, j_start - 1]):
            q_out = (q_in[0, 1, 0] + q_in[-1, 1, 0] + q_in[0, 0, 0]) * third

        with horizontal(region[i_end, j_start]):
            q_out = (q_in[0, 0, 0] + q_in[1, 0, 0] + q_in[0, -1, 0]) * third
        with horizontal(region[i_end + 1, j_start]):
            q_out = (q_in[-1, 0, 0] + q_in[0, 0, 0] + q_in[-1, -1, 0]) * third
        with horizontal(region[i_end, j_start - 1]):
            q_out = (q_in[0, 1, 0] + q_in[1, 1, 0] + q_in[0, 0, 0]) * third

        with horizontal(region[i_end, j_end]):
            q_out = (q_in[0, 0, 0] + q_in[1, 0, 0] + q_in[0, 1, 0]) * third
        with horizontal(region[i_end + 1, j_end]):
            q_out = (q_in[-1, 0, 0] + q_in[0, 0, 0] + q_in[-1, 1, 0]) * third
        with horizontal(region[i_end, j_end + 1]):
            q_out = (q_in[0, -1, 0] + q_in[1, -1, 0] + q_in[0, 0, 0]) * third

        with horizontal(region[i_start, j_end]):
            q_out = (q_in[0, 0, 0] + q_in[-1, 0, 0] + q_in[0, 1, 0]) * third
        with horizontal(region[i_start - 1, j_end]):
            q_out = (q_in[1, 0, 0] + q_in[0, 0, 0] + q_in[1, 1, 0]) * third
        with horizontal(region[i_start, j_end + 1]):
            q_out = (q_in[0, -1, 0] + q_in[-1, -1, 0] + q_in[0, 0, 0]) * third


class HyperdiffusionDamping:
    """
    Fortran name is del2_cubed
    """

    def __init__(
        self,
        grid_indexing: GridIndexing,
        damping_coefficients: DampingCoefficients,
        rarea,
        nmax: int,
    ):
        """
        Args:
            grid: fv3core grid object
        """
        self._del6_u = damping_coefficients.del6_u
        self._del6_v = damping_coefficients.del6_v
        self._rarea = rarea
        origin = grid_indexing.origin_full()
        domain = grid_indexing.domain_full()
        ax_offsets = axis_offsets(spec.grid, origin, domain)
        self._fx = utils.make_storage_from_shape(grid_indexing.max_shape)
        self._fy = utils.make_storage_from_shape(grid_indexing.max_shape)
        self._q = utils.make_storage_from_shape(grid_indexing.max_shape)

        self._corner_fill = FrozenStencil(
            func=corner_fill, origin=origin, domain=domain, externals=ax_offsets
        )

        self._copy_stencil = FrozenStencil(func=copy_defn, origin=origin, domain=domain)

        self._ntimes = min(3, nmax)
        origins = []
        domains_x = []
        domains_y = []
        domains = []
        for n_halo in range(self._ntimes - 1, -1, -1):
            origin, domain = grid_indexing.get_origin_domain(
                [X_DIM, Y_DIM, Z_DIM], halos=(n_halo, n_halo)
            )
            _, domain_x = grid_indexing.get_origin_domain(
                [X_INTERFACE_DIM, Y_DIM, Z_DIM], halos=(n_halo, n_halo)
            )
            _, domain_y = grid_indexing.get_origin_domain(
                [X_DIM, Y_INTERFACE_DIM, Z_DIM], halos=(n_halo, n_halo)
            )
            origins.append(origin)
            domains.append(domain)
            domains_x.append(domain_x)
            domains_y.append(domain_y)

        self._compute_zonal_flux = get_stencils_with_varied_bounds(
            compute_zonal_flux,
            origins,
            domains_x,
        )
        self._compute_meridional_flux = get_stencils_with_varied_bounds(
            compute_meridional_flux,
            origins,
            domains_y,
        )
        self._update_q = get_stencils_with_varied_bounds(
            update_q,
            origins,
            domains,
        )

        self._copy_corners_x: corners.CopyCorners = corners.CopyCorners("x")
        """Stencil responsible for doing corners updates in x-direction."""
        self._copy_corners_y: corners.CopyCorners = corners.CopyCorners("y")
        """Stencil responsible for doing corners updates in y-direction."""

    def __call__(self, qdel: FloatField, cd: float):
        """
        Perform hyperdiffusion damping/filtering

        Args:
            qdel (inout): Variable to be filterd
            nmax: Number of times to apply filtering
            cd: Damping coeffcient
        """

        for n in range(self._ntimes):
            nt = self._ntimes - (n + 1)

            # Fill in appropriate corner values
            self._corner_fill(qdel, self._q)

            if nt > 0:
                self._copy_corners_x(self._q)

            self._compute_zonal_flux[n](self._fx, self._q, self._del6_v)

            if nt > 0:
                self._copy_corners_y(self._q)

            self._compute_meridional_flux[n](self._fy, self._q, self._del6_u)

            self._copy_stencil(self._q, qdel)

            # Update q values
            self._update_q[n](qdel, self._rarea, self._fx, self._fy, cd)
