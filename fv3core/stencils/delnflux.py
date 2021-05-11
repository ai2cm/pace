from typing import Optional

import gt4py.gtscript as gtscript
import numpy as np
from gt4py.gtscript import (
    __INLINED,
    FORWARD,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)

import fv3core._config as spec
import fv3core.utils.corners as corners
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil, gtstencil
from fv3core.utils.grid import axis_offsets
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK


def calc_damp(damp4: FloatField, nord: FloatFieldK, damp_c: FloatFieldK, da_min: float):
    with computation(FORWARD), interval(...):
        damp4 = (damp_c * da_min) ** (nord + 1)


def fx_calc_stencil_nord(q: FloatField, del6_v: FloatFieldIJ, fx: FloatField):
    from __externals__ import (
        local_ie,
        local_is,
        local_je,
        local_js,
        nord0,
        nord1,
        nord2,
        nord3,
    )

    with computation(PARALLEL), interval(0, 1):
        if __INLINED(nord0 == 0):
            with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
                fx = fx_calculation(q, del6_v)
        else:
            fx = fx_calculation(q, del6_v)
    with computation(PARALLEL), interval(1, 2):
        if __INLINED(nord1 == 0):
            with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
                fx = fx_calculation(q, del6_v)
        else:
            fx = fx_calculation(q, del6_v)
    with computation(PARALLEL), interval(2, 3):
        if __INLINED(nord2 == 0):
            with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
                fx = fx_calculation(q, del6_v)
        else:
            fx = fx_calculation(q, del6_v)
    with computation(PARALLEL), interval(3, None):
        if __INLINED(nord3 == 0):
            with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
                fx = fx_calculation(q, del6_v)
        else:
            fx = fx_calculation(q, del6_v)


def fy_calc_stencil_nord(q: FloatField, del6_u: FloatFieldIJ, fy: FloatField):
    from __externals__ import (
        local_ie,
        local_is,
        local_je,
        local_js,
        nord0,
        nord1,
        nord2,
        nord3,
    )

    with computation(PARALLEL), interval(0, 1):
        if __INLINED(nord0 == 0):
            with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
                fy = fy_calculation(q, del6_u)
        else:
            fy = fy_calculation(q, del6_u)
    with computation(PARALLEL), interval(1, 2):
        if __INLINED(nord1 == 0):
            with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
                fy = fy_calculation(q, del6_u)
        else:
            fy = fy_calculation(q, del6_u)
    with computation(PARALLEL), interval(2, 3):
        if __INLINED(nord2 == 0):
            with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
                fy = fy_calculation(q, del6_u)
        else:
            fy = fy_calculation(q, del6_u)
    with computation(PARALLEL), interval(3, None):
        if __INLINED(nord3 == 0):
            with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
                fy = fy_calculation(q, del6_u)
        else:
            fy = fy_calculation(q, del6_u)


def fx_calc_stencil_column(q: FloatField, del6_v: FloatFieldIJ, fx: FloatField):
    from __externals__ import nord0, nord1, nord2, nord3

    with computation(PARALLEL), interval(0, 1):
        if __INLINED(nord0 > 0):
            fx = fx_calculation_neg(q, del6_v)
    with computation(PARALLEL), interval(1, 2):
        if __INLINED(nord1 > 0):
            fx = fx_calculation_neg(q, del6_v)
    with computation(PARALLEL), interval(2, 3):
        if __INLINED(nord2 > 0):
            fx = fx_calculation_neg(q, del6_v)
    with computation(PARALLEL), interval(3, None):
        if __INLINED(nord3 > 0):
            fx = fx_calculation_neg(q, del6_v)


def fy_calc_stencil_column(q: FloatField, del6_u: FloatFieldIJ, fy: FloatField):
    from __externals__ import nord0, nord1, nord2, nord3

    with computation(PARALLEL), interval(0, 1):
        if __INLINED(nord0 > 0):
            fy = fy_calculation_neg(q, del6_u)
    with computation(PARALLEL), interval(1, 2):
        if __INLINED(nord1 > 0):
            fy = fy_calculation_neg(q, del6_u)
    with computation(PARALLEL), interval(2, 3):
        if __INLINED(nord2 > 0):
            fy = fy_calculation_neg(q, del6_u)
    with computation(PARALLEL), interval(3, None):
        if __INLINED(nord3 > 0):
            fy = fy_calculation_neg(q, del6_u)


@gtscript.function
def fx_calculation(q: FloatField, del6_v: FloatField):
    return del6_v * (q[-1, 0, 0] - q)


@gtscript.function
def fx_calculation_neg(q: FloatField, del6_v: FloatField):
    return -del6_v * (q[-1, 0, 0] - q)


@gtscript.function
def fy_calculation(q: FloatField, del6_u: FloatField):
    return del6_u * (q[0, -1, 0] - q)


@gtscript.function
def fy_calculation_neg(q: FloatField, del6_u: FloatField):
    return -del6_u * (q[0, -1, 0] - q)


# WARNING: untested
@gtstencil
def fx_firstorder_use_sg(
    q: FloatField,
    sin_sg1: FloatField,
    sin_sg3: FloatField,
    dy: FloatField,
    rdxc: FloatField,
    fx: FloatField,
):
    with computation(PARALLEL), interval(...):
        fx = 0.5 * (sin_sg3[-1, 0, 0] + sin_sg1) * dy * (q[-1, 0, 0] - q) * rdxc


# WARNING: untested
@gtstencil
def fy_firstorder_use_sg(
    q: FloatField,
    sin_sg2: FloatField,
    sin_sg4: FloatField,
    dx: FloatField,
    rdyc: FloatField,
    fy: FloatField,
):
    with computation(PARALLEL), interval(...):
        fy = 0.5 * (sin_sg4[0, -1, 0] + sin_sg2) * dx * (q[0, -1, 0] - q) * rdyc


def d2_highorder_stencil(
    fx: FloatField, fy: FloatField, rarea: FloatFieldIJ, d2: FloatField
):
    from __externals__ import nord0, nord1, nord2, nord3

    with computation(PARALLEL), interval(0, 1):
        if __INLINED(nord0 > 0):
            d2 = d2_highorder(fx, fy, rarea)
    with computation(PARALLEL), interval(1, 2):
        if __INLINED(nord1 > 0):
            d2 = d2_highorder(fx, fy, rarea)
    with computation(PARALLEL), interval(2, 3):
        if __INLINED(nord2 > 0):
            d2 = d2_highorder(fx, fy, rarea)
    with computation(PARALLEL), interval(3, None):
        if __INLINED(nord3 > 0):
            d2 = d2_highorder(fx, fy, rarea)


@gtscript.function
def d2_highorder(fx: FloatField, fy: FloatField, rarea: FloatField):
    d2 = (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea
    return d2


def d2_damp_interval(q: FloatField, d2: FloatField, damp: FloatFieldK):
    from __externals__ import (
        local_ie,
        local_is,
        local_je,
        local_js,
        nord0,
        nord1,
        nord2,
        nord3,
    )

    with computation(PARALLEL), interval(0, 1):
        if __INLINED(nord0 == 0):
            with horizontal(
                region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
            ):
                d2 = damp * q
        else:
            d2 = damp * q
    with computation(PARALLEL), interval(1, 2):
        if __INLINED(nord1 == 0):
            with horizontal(
                region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
            ):
                d2 = damp * q
        else:
            d2 = damp * q
    with computation(PARALLEL), interval(2, 3):
        if __INLINED(nord2 == 0):
            with horizontal(
                region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
            ):
                d2 = damp * q
        else:
            d2 = damp * q
    with computation(PARALLEL), interval(3, None):
        if __INLINED(nord3 == 0):
            with horizontal(
                region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
            ):
                d2 = damp * q
        else:
            d2 = damp * q


def copy_stencil_interval(q_in: FloatField, q_out: FloatField):
    from __externals__ import (
        local_ie,
        local_is,
        local_je,
        local_js,
        nord0,
        nord1,
        nord2,
        nord3,
    )

    with computation(PARALLEL), interval(0, 1):
        if __INLINED(nord0 == 0):
            with horizontal(
                region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
            ):
                q_out = q_in
        else:
            q_out = q_in
    with computation(PARALLEL), interval(1, 2):
        if __INLINED(nord1 == 0):
            with horizontal(
                region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
            ):
                q_out = q_in
        else:
            q_out = q_in
    with computation(PARALLEL), interval(2, 3):
        if __INLINED(nord2 == 0):
            with horizontal(
                region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
            ):
                q_out = q_in
        else:
            q_out = q_in
    with computation(PARALLEL), interval(3, None):
        if __INLINED(nord3 == 0):
            with horizontal(
                region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
            ):
                q_out = q_in
        else:
            q_out = q_in


def add_diffusive_component(
    fx: FloatField, fx2: FloatField, fy: FloatField, fy2: FloatField
):
    with computation(PARALLEL), interval(...):
        fx = fx + fx2
        fy = fy + fy2


def diffusive_damp(
    fx: FloatField,
    fx2: FloatField,
    fy: FloatField,
    fy2: FloatField,
    mass: FloatField,
    damp: FloatFieldK,
):
    with computation(PARALLEL), interval(...):
        fx = fx + 0.5 * damp * (mass[-1, 0, 0] + mass) * fx2
        fy = fy + 0.5 * damp * (mass[0, -1, 0] + mass) * fy2


class DelnFlux:
    """
    Fortran name is deln_flux
    The test class is DelnFlux
    """

    def __init__(self, nord: FloatFieldK, damp_c: FloatFieldK):
        """
        nord sets the order of damping to apply:
        nord = 0:   del-2
        nord = 1:   del-4
        nord = 2:   del-6

        nord and damp_c define the damping coefficient used in DelnFluxNoSG
        """
        self._no_compute = False
        if (damp_c <= 1e-4).all():
            self._no_compute = True
        elif (damp_c[:-1] <= 1e-4).any():
            raise NotImplementedError(
                "damp_c currently must be always greater than 10^-4 for delnflux"
            )

        grid = spec.grid
        nk = grid.npz
        self._origin = (grid.isd, grid.jsd, 0)

        shape = grid.domain_shape_full(add=(1, 1, 1))
        k_shape = (1, 1, nk)

        self._damp_3d = utils.make_storage_from_shape(k_shape)
        # fields must be 3d to assign to them
        self._fx2 = utils.make_storage_from_shape(shape)
        self._fy2 = utils.make_storage_from_shape(shape)
        self._d2 = utils.make_storage_from_shape(grid.domain_shape_full())

        diffuse_origin = (grid.is_, grid.js, 0)
        extended_domain = (grid.nic + 1, grid.njc + 1, nk)

        self._damping_factor_calculation = FrozenStencil(
            calc_damp, origin=(0, 0, 0), domain=k_shape
        )
        self._add_diffusive_stencil = FrozenStencil(
            add_diffusive_component, origin=diffuse_origin, domain=extended_domain
        )
        self._diffusive_damp_stencil = FrozenStencil(
            diffusive_damp, origin=diffuse_origin, domain=extended_domain
        )

        self._damping_factor_calculation(self._damp_3d, nord, damp_c, grid.da_min)
        self._damp = utils.make_storage_data(self._damp_3d[0, 0, :], (nk,), (0,))

        self.delnflux_nosg = DelnFluxNoSG(nord, nk=nk)

    def __call__(
        self,
        q: FloatField,
        fx: FloatField,
        fy: FloatField,
        d2: Optional["FloatField"] = None,
        mass: Optional["FloatField"] = None,
    ):
        """
        Del-n damping for fluxes, where n = 2 * nord + 2
        Args:
            q: Field for which to calculate damped fluxes (in)
            fx: x-flux on A-grid (inout)
            fy: y-flux on A-grid (inout)
            d2: A damped copy of the q field (in)
            mass: Mass to weight the diffusive flux by (in)
        """
        if self._no_compute is True:
            return fx, fy

        if d2 is None:
            d2 = self._d2

        self.delnflux_nosg(q, self._fx2, self._fy2, self._damp, d2, mass)

        if mass is None:
            self._add_diffusive_stencil(fx, self._fx2, fy, self._fy2)
        else:
            # TODO: To join these stencils you need to overcompute, making the edges
            # 'wrong', but not actually used, separating now for comparison sanity.

            # diffusive_damp(fx, fx2, fy, fy2, mass, damp, origin=diffuse_origin,
            # domain=(grid.nic + 1, grid.njc + 1, nk))
            self._diffusive_damp_stencil(fx, self._fx2, fy, self._fy2, mass, self._damp)

        return fx, fy


class DelnFluxNoSG:
    """
    This contains the mechanics of del6_vt and some of deln_flux from
    the Fortran code, since they are very similar routines. The test class
    is Del6VtFlux
    """

    def __init__(self, nord, nk: Optional[int] = None):
        """
        nord sets the order of damping to apply:
        nord = 0:   del-2
        nord = 1:   del-4
        nord = 2:   del-6
        """
        if max(nord[:]) > 3:
            raise ValueError("nord must be less than 3")
        if not np.all(n in [0, 2, 3] for n in nord[:]):
            raise NotImplementedError("nord must have values 0, 2, or 3")
        self._nmax = int(max(nord[:]))
        self._grid = spec.grid
        i1 = self._grid.is_ - 1 - self._nmax
        i2 = self._grid.ie + 1 + self._nmax
        j1 = self._grid.js - 1 - self._nmax
        j2 = self._grid.je + 1 + self._nmax
        if nk is None:
            nk = self._grid.npz
        self._nk = nk
        origin_d2 = (i1, j1, 0)
        domain_d2 = (i2 - i1 + 1, j2 - j1 + 1, self._nk)
        f1_ny = self._grid.je - self._grid.js + 1 + 2 * self._nmax
        f1_nx = self._grid.ie - self._grid.is_ + 2 + 2 * self._nmax
        fx_origin = (self._grid.is_ - self._nmax, self._grid.js - self._nmax, 0)
        self._nord = nord

        if self._nk <= 3:
            raise Exception("nk must be more than 3 for DelnFluxNoSG")
        self._k_bounds = [1, 1, 1, self._nk - 3]

        preamble_ax_offsets = axis_offsets(self._grid, origin_d2, domain_d2)
        fx_ax_offsets = axis_offsets(self._grid, fx_origin, (f1_nx, f1_ny, self._nk))
        fy_ax_offsets = axis_offsets(
            self._grid, fx_origin, (f1_nx - 1, f1_ny + 1, self._nk)
        )

        nord_dictionary = {
            "nord0": nord[0],
            "nord1": nord[1],
            "nord2": nord[2],
            "nord3": nord[3],
        }

        self._d2_damp = FrozenStencil(
            d2_damp_interval,
            externals={
                **nord_dictionary,
                **preamble_ax_offsets,
            },
            origin=origin_d2,
            domain=domain_d2,
        )

        self._copy_stencil_interval = FrozenStencil(
            copy_stencil_interval,
            externals={
                **nord_dictionary,
                **preamble_ax_offsets,
            },
            origin=origin_d2,
            domain=domain_d2,
        )

        self._d2_stencil = gtstencil(
            d2_highorder_stencil,
            externals={**nord_dictionary},
        )
        self._column_conditional_fx_calculation = gtstencil(
            fx_calc_stencil_column,
            externals={**nord_dictionary},
        )
        self._column_conditional_fy_calculation = gtstencil(
            fy_calc_stencil_column,
            externals={**nord_dictionary},
        )
        self._fx_calc_stencil = FrozenStencil(
            fx_calc_stencil_nord,
            externals={**fx_ax_offsets, **nord_dictionary},
            origin=fx_origin,
            domain=(f1_nx, f1_ny, self._nk),
        )
        self._fy_calc_stencil = FrozenStencil(
            fy_calc_stencil_nord,
            externals={**fy_ax_offsets, **nord_dictionary},
            origin=fx_origin,
            domain=(f1_nx - 1, f1_ny + 1, self._nk),
        )

    def __call__(self, q, fx2, fy2, damp_c, d2, mass=None):
        """
        Applies del-n damping to fluxes, where n is set by nord.

        Args:
            q: Field for which to calculate damped fluxes (in)
            fx2: diffusive x-flux on A grid (in/out)
            fy2: diffusive y-flux on A grid (in/out)
            damp_c: damping coefficient for q (in)
            d2: A damped copy of the q field (in)
            mass: Mass to weight the diffusive flux by (in)
        """

        if mass is None:
            self._d2_damp(q, d2, damp_c)
        else:
            self._copy_stencil_interval(q, d2)

        for kstart, k_range in enumerate(self._k_bounds):
            if self._nord[kstart] > 0:
                corners.copy_corners_x_stencil(
                    d2,
                    origin=(self._grid.isd, self._grid.jsd, kstart),
                    domain=(self._grid.nid, self._grid.njd, k_range),
                )

        self._fx_calc_stencil(d2, self._grid.del6_v, fx2)

        for kstart, k_range in enumerate(self._k_bounds):
            if self._nord[kstart] > 0:
                corners.copy_corners_y_stencil(
                    d2,
                    origin=(self._grid.isd, self._grid.jsd, kstart),
                    domain=(self._grid.nid, self._grid.njd, k_range),
                )

        self._fy_calc_stencil(d2, self._grid.del6_u, fy2)

        for n in range(self._nmax):
            nt = self._nmax - 1 - n
            nt_origin = (self._grid.is_ - nt - 1, self._grid.js - nt - 1, 0)
            nt_ny = self._grid.je - self._grid.js + 3 + 2 * nt
            nt_nx = self._grid.ie - self._grid.is_ + 3 + 2 * nt
            self._d2_stencil(
                fx2,
                fy2,
                self._grid.rarea,
                d2,
                origin=nt_origin,
                domain=(nt_nx, nt_ny, self._nk),
            )

            for kstart, k_range in enumerate(self._k_bounds):
                if self._nord[kstart] > 0:
                    corners.copy_corners_x_stencil(
                        d2,
                        origin=(self._grid.isd, self._grid.jsd, kstart),
                        domain=(self._grid.nid, self._grid.njd, k_range),
                    )

            nt_origin = (self._grid.is_ - nt, self._grid.js - nt, 0)
            self._column_conditional_fx_calculation(
                d2,
                self._grid.del6_v,
                fx2,
                origin=nt_origin,
                domain=(nt_nx - 1, nt_ny - 2, self._nk),
            )

            for kstart, k_range in enumerate(self._k_bounds):
                if self._nord[kstart] > 0:
                    corners.copy_corners_y_stencil(
                        d2,
                        origin=(self._grid.isd, self._grid.jsd, kstart),
                        domain=(self._grid.nid, self._grid.njd, k_range),
                    )

            self._column_conditional_fy_calculation(
                d2,
                self._grid.del6_u,
                fy2,
                origin=nt_origin,
                domain=(nt_nx - 2, nt_ny - 1, self._nk),
            )
