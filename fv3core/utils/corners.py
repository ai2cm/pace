from gt4py import gtscript
from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil, gtstencil
from fv3core.stencils.basic_operations import copy_defn
from fv3core.utils.grid import axis_offsets
from fv3core.utils.typing import FloatField


class CopyCorners:
    """
    Helper-class to copy corners corresponding to the fortran functions
    copy_corners_x or copy_corners_y respectively
    """

    def __init__(self, direction: str, temporary_field=None) -> None:
        self.grid = spec.grid
        """The grid for this stencil"""

        origin = self.grid.full_origin()
        """The origin for the corner computation"""

        domain = self.grid.domain_shape_full(add=(0, 0, 1))
        """The full domain required to do corner computation everywhere"""

        if temporary_field is not None:
            self._corner_tmp = temporary_field
        else:
            self._corner_tmp = utils.make_storage_from_shape(
                self.grid.domain_shape_full(add=(1, 1, 1)), origin=origin
            )

        self._copy_full_domain = FrozenStencil(
            func=copy_defn,
            origin=origin,
            domain=domain,
        )
        """Stencil Wrapper to do the copy of the input field to the temporary field"""

        ax_offsets = axis_offsets(spec.grid, origin, domain)
        if direction == "x":
            self._copy_corners = FrozenStencil(
                func=copy_corners_x_stencil_defn,
                origin=origin,
                domain=domain,
                externals={
                    **ax_offsets,
                },
            )
        elif direction == "y":
            self._copy_corners = FrozenStencil(
                func=copy_corners_y_stencil_defn,
                origin=origin,
                domain=domain,
                externals={
                    **ax_offsets,
                },
            )
        else:
            raise ValueError("Direction must be either 'x' or 'y'")

    def __call__(self, field: FloatField):
        """
        Fills cell quantity field using corners from itself and multipliers
        in the dirction specified initialization of the instance of this class.
        """
        self._copy_full_domain(field, self._corner_tmp)
        self._copy_corners(self._corner_tmp, field)


@gtscript.function
def fill_corners_2cells_mult_x(
    q: FloatField,
    q_corner: FloatField,
    sw_mult: float,
    se_mult: float,
    nw_mult: float,
    ne_mult: float,
):
    """
    Fills cell quantity q using corners from q_corner and multipliers in x-dir.
    """
    from __externals__ import i_end, i_start, j_end, j_start

    # Southwest
    with horizontal(region[i_start - 1, j_start - 1]):
        q = sw_mult * q_corner[0, 1, 0]
    with horizontal(region[i_start - 2, j_start - 1]):
        q = sw_mult * q_corner[1, 2, 0]

    # Southeast
    with horizontal(region[i_end + 1, j_start - 1]):
        q = se_mult * q_corner[0, 1, 0]
    with horizontal(region[i_end + 2, j_start - 1]):
        q = se_mult * q_corner[-1, 2, 0]

    # Northwest
    with horizontal(region[i_start - 1, j_end + 1]):
        q = nw_mult * q_corner[0, -1, 0]
    with horizontal(region[i_start - 2, j_end + 1]):
        q = nw_mult * q_corner[1, -2, 0]

    # Northeast
    with horizontal(region[i_end + 1, j_end + 1]):
        q = ne_mult * q_corner[0, -1, 0]
    with horizontal(region[i_end + 2, j_end + 1]):
        q = ne_mult * q_corner[-1, -2, 0]

    return q


@gtstencil
def fill_corners_2cells_x_stencil(q: FloatField):
    with computation(PARALLEL), interval(...):
        q = fill_corners_2cells_mult_x(q, q, 1.0, 1.0, 1.0, 1.0)


@gtstencil
def fill_corners_2cells_y_stencil(q: FloatField):
    with computation(PARALLEL), interval(...):
        q = fill_corners_2cells_mult_y(q, q, 1.0, 1.0, 1.0, 1.0)


@gtscript.function
def fill_corners_2cells_x(q: FloatField):
    """
    Fills cell quantity q in x-dir.
    """
    return fill_corners_2cells_mult_x(q, q, 1.0, 1.0, 1.0, 1.0)


@gtscript.function
def fill_corners_3cells_mult_x(
    q: FloatField,
    q_corner: FloatField,
    sw_mult: float,
    se_mult: float,
    nw_mult: float,
    ne_mult: float,
):
    """
    Fills cell quantity q using corners from q_corner and multipliers in x-dir.
    """
    from __externals__ import i_end, i_start, j_end, j_start

    q = fill_corners_2cells_mult_x(q, q_corner, sw_mult, se_mult, nw_mult, ne_mult)

    # Southwest
    with horizontal(region[i_start - 3, j_start - 1]):
        q = sw_mult * q_corner[2, 3, 0]

    # Southeast
    with horizontal(region[i_end + 3, j_start - 1]):
        q = se_mult * q_corner[-2, 3, 0]

    # Northwest
    with horizontal(region[i_start - 3, j_end + 1]):
        q = nw_mult * q_corner[2, -3, 0]

    # Northeast
    with horizontal(region[i_end + 3, j_end + 1]):
        q = ne_mult * q_corner[-2, -3, 0]

    return q


@gtscript.function
def fill_corners_2cells_mult_y(
    q: FloatField,
    q_corner: FloatField,
    sw_mult: float,
    se_mult: float,
    nw_mult: float,
    ne_mult: float,
):
    """
    Fills cell quantity q using corners from q_corner and multipliers in y-dir.
    """
    from __externals__ import i_end, i_start, j_end, j_start

    # Southwest
    with horizontal(region[i_start - 1, j_start - 1]):
        q = sw_mult * q_corner[1, 0, 0]
    with horizontal(region[i_start - 1, j_start - 2]):
        q = sw_mult * q_corner[2, 1, 0]

    # Southeast
    with horizontal(region[i_end + 1, j_start - 1]):
        q = se_mult * q_corner[-1, 0, 0]
    with horizontal(region[i_end + 1, j_start - 2]):
        q = se_mult * q_corner[-2, 1, 0]

    # Northwest
    with horizontal(region[i_start - 1, j_end + 1]):
        q = nw_mult * q_corner[1, 0, 0]
    with horizontal(region[i_start - 1, j_end + 2]):
        q = nw_mult * q_corner[2, -1, 0]

    # Northeast
    with horizontal(region[i_end + 1, j_end + 1]):
        q = ne_mult * q_corner[-1, 0, 0]
    with horizontal(region[i_end + 1, j_end + 2]):
        q = ne_mult * q_corner[-2, -1, 0]

    return q


@gtscript.function
def fill_corners_2cells_y(q: FloatField):
    """
    Fills cell quantity q in y-dir.
    """
    return fill_corners_2cells_mult_y(q, q, 1.0, 1.0, 1.0, 1.0)


@gtscript.function
def fill_corners_3cells_mult_y(
    q: FloatField,
    q_corner: FloatField,
    sw_mult: float,
    se_mult: float,
    nw_mult: float,
    ne_mult: float,
):
    """
    Fills cell quantity q using corners from q_corner and multipliers in y-dir.
    """
    from __externals__ import i_end, i_start, j_end, j_start

    q = fill_corners_2cells_mult_y(q, q_corner, sw_mult, se_mult, nw_mult, ne_mult)

    # Southwest
    with horizontal(region[i_start - 1, j_start - 3]):
        q = sw_mult * q_corner[3, 2, 0]

    # Southeast
    with horizontal(region[i_end + 1, j_start - 3]):
        q = se_mult * q_corner[-3, 2, 0]

    # Northwest
    with horizontal(region[i_start - 1, j_end + 3]):
        q = nw_mult * q_corner[3, -2, 0]

    # Northeast
    with horizontal(region[i_end + 1, j_end + 3]):
        q = ne_mult * q_corner[-3, -2, 0]

    return q


def fill_corners_cells(q: FloatField, direction: str, num_fill: int = 2):
    """
    Fill corners of q from Python.

    Corresponds to fill4corners in Fortran.

    Args:
        q (inout): Cell field
        direction: Direction to fill. Either "x" or "y".
        num_fill: Number of indices to fill
    """

    def definition(q: FloatField):
        from __externals__ import func

        with computation(PARALLEL), interval(...):
            q = func(q, q, 1.0, 1.0, 1.0, 1.0)

    if num_fill not in (2, 3):
        raise ValueError("Only supports 2 <= num_fill <= 3")

    if direction == "x":
        func = (
            fill_corners_2cells_mult_x if num_fill == 2 else fill_corners_3cells_mult_x
        )
        stencil = gtstencil(definition, externals={"func": func})
    elif direction == "y":
        func = (
            fill_corners_2cells_mult_y if num_fill == 2 else fill_corners_3cells_mult_y
        )
        stencil = gtstencil(definition, externals={"func": func})
    else:
        raise ValueError("Direction not recognized. Specify either x or y")

    extent = 3
    origin = (spec.grid.is_ - extent, spec.grid.js - extent, 0)
    domain = (spec.grid.nic + 2 * extent, spec.grid.njc + 2 * extent, q.shape[2])
    stencil(q, origin=origin, domain=domain)


def copy_corners_x_stencil_defn(q_in: FloatField, q_out: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        with horizontal(
            region[i_start - 3, j_start - 3], region[i_end + 3, j_start - 3]
        ):
            q_out = q_in[0, 5, 0]
        with horizontal(
            region[i_start - 2, j_start - 3], region[i_end + 3, j_start - 2]
        ):
            q_out = q_in[-1, 4, 0]
        with horizontal(
            region[i_start - 1, j_start - 3], region[i_end + 3, j_start - 1]
        ):
            q_out = q_in[-2, 3, 0]
        with horizontal(
            region[i_start - 3, j_start - 2], region[i_end + 2, j_start - 3]
        ):
            q_out = q_in[1, 4, 0]
        with horizontal(
            region[i_start - 2, j_start - 2], region[i_end + 2, j_start - 2]
        ):
            q_out = q_in[0, 3, 0]
        with horizontal(
            region[i_start - 1, j_start - 2], region[i_end + 2, j_start - 1]
        ):
            q_out = q_in[-1, 2, 0]
        with horizontal(
            region[i_start - 3, j_start - 1], region[i_end + 1, j_start - 3]
        ):
            q_out = q_in[2, 3, 0]
        with horizontal(
            region[i_start - 2, j_start - 1], region[i_end + 1, j_start - 2]
        ):
            q_out = q_in[1, 2, 0]
        with horizontal(
            region[i_start - 1, j_start - 1], region[i_end + 1, j_start - 1]
        ):
            q_out = q_in[0, 1, 0]
        with horizontal(region[i_start - 3, j_end + 1], region[i_end + 1, j_end + 3]):
            q_out = q_in[2, -3, 0]
        with horizontal(region[i_start - 2, j_end + 1], region[i_end + 1, j_end + 2]):
            q_out = q_in[1, -2, 0]
        with horizontal(region[i_start - 1, j_end + 1], region[i_end + 1, j_end + 1]):
            q_out = q_in[0, -1, 0]
        with horizontal(region[i_start - 3, j_end + 2], region[i_end + 2, j_end + 3]):
            q_out = q_in[1, -4, 0]
        with horizontal(region[i_start - 2, j_end + 2], region[i_end + 2, j_end + 2]):
            q_out = q_in[0, -3, 0]
        with horizontal(region[i_start - 1, j_end + 2], region[i_end + 2, j_end + 1]):
            q_out = q_in[-1, -2, 0]
        with horizontal(region[i_start - 3, j_end + 3], region[i_end + 3, j_end + 3]):
            q_out = q_in[0, -5, 0]
        with horizontal(region[i_start - 2, j_end + 3], region[i_end + 3, j_end + 2]):
            q_out = q_in[-1, -4, 0]
        with horizontal(region[i_start - 1, j_end + 3], region[i_end + 3, j_end + 1]):
            q_out = q_in[-2, -3, 0]


def copy_corners_y_stencil_defn(q_in: FloatField, q_out: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        with horizontal(
            region[i_start - 3, j_start - 3], region[i_start - 3, j_end + 3]
        ):
            q_out = q_in[5, 0, 0]
        with horizontal(
            region[i_start - 2, j_start - 3], region[i_start - 3, j_end + 2]
        ):
            q_out = q_in[4, 1, 0]
        with horizontal(
            region[i_start - 1, j_start - 3], region[i_start - 3, j_end + 1]
        ):
            q_out = q_in[3, 2, 0]
        with horizontal(
            region[i_start - 3, j_start - 2], region[i_start - 2, j_end + 3]
        ):
            q_out = q_in[4, -1, 0]
        with horizontal(
            region[i_start - 2, j_start - 2], region[i_start - 2, j_end + 2]
        ):
            q_out = q_in[3, 0, 0]
        with horizontal(
            region[i_start - 1, j_start - 2], region[i_start - 2, j_end + 1]
        ):
            q_out = q_in[2, 1, 0]
        with horizontal(
            region[i_start - 3, j_start - 1], region[i_start - 1, j_end + 3]
        ):
            q_out = q_in[3, -2, 0]
        with horizontal(
            region[i_start - 2, j_start - 1], region[i_start - 1, j_end + 2]
        ):
            q_out = q_in[2, -1, 0]
        with horizontal(
            region[i_start - 1, j_start - 1], region[i_start - 1, j_end + 1]
        ):
            q_out = q_in[1, 0, 0]
        with horizontal(region[i_end + 1, j_start - 3], region[i_end + 3, j_end + 1]):
            q_out = q_in[-3, 2, 0]
        with horizontal(region[i_end + 2, j_start - 3], region[i_end + 3, j_end + 2]):
            q_out = q_in[-4, 1, 0]
        with horizontal(region[i_end + 3, j_start - 3], region[i_end + 3, j_end + 3]):
            q_out = q_in[-5, 0, 0]
        with horizontal(region[i_end + 1, j_start - 2], region[i_end + 2, j_end + 1]):
            q_out = q_in[-2, 1, 0]
        with horizontal(region[i_end + 2, j_start - 2], region[i_end + 2, j_end + 2]):
            q_out = q_in[-3, 0, 0]
        with horizontal(region[i_end + 3, j_start - 2], region[i_end + 2, j_end + 3]):
            q_out = q_in[-4, -1, 0]
        with horizontal(region[i_end + 1, j_start - 1], region[i_end + 1, j_end + 1]):
            q_out = q_in[-1, 0, 0]
        with horizontal(region[i_end + 2, j_start - 1], region[i_end + 1, j_end + 2]):
            q_out = q_in[-2, -1, 0]
        with horizontal(region[i_end + 3, j_start - 1], region[i_end + 1, j_end + 3]):
            q_out = q_in[-3, -2, 0]


class FillCornersBGrid:
    """
    Helper-class to fill corners corresponding to the fortran function
    fill_corners with BGRID=.true. and either FILL=YDir or FILL=YDIR
    """

    def __init__(
        self, direction: str, temporary_field=None, origin=None, domain=None
    ) -> None:
        self.grid = spec.grid
        """The grid for this stencil"""
        if origin is None:
            origin = self.grid.full_origin()
        """The origin for the corner computation"""
        if domain is None:
            domain = self.grid.domain_shape_full()
        """The full domain required to do corner computation everywhere"""

        if temporary_field is not None:
            self._corner_tmp = temporary_field
        else:
            self._corner_tmp = utils.make_storage_from_shape(
                self.grid.domain_shape_full(add=(1, 1, 1)), origin=origin
            )

        self._copy_full_domain = FrozenStencil(
            func=copy_defn,
            origin=origin,
            domain=domain,
        )

        """Stencil Wrapper to do the copy of the input field to the temporary field"""

        ax_offsets = axis_offsets(self.grid, origin, domain)

        if direction == "x":
            self._fill_corners_bgrid = FrozenStencil(
                func=fill_corners_bgrid_x_defn,
                origin=origin,
                domain=domain,
                externals=ax_offsets,
            )
        elif direction == "y":
            self._fill_corners_bgrid = FrozenStencil(
                func=fill_corners_bgrid_y_defn,
                origin=origin,
                domain=domain,
                externals=ax_offsets,
            )

        else:
            raise ValueError("Direction must be either 'x' or 'y'")

    def __call__(self, field: FloatField):
        self._copy_full_domain(field, self._corner_tmp)
        self._fill_corners_bgrid(self._corner_tmp, field)


def fill_corners_bgrid_x_defn(q_in: FloatField, q_out: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        # sw and se corner
        with horizontal(
            region[i_start - 1, j_start - 1], region[i_end + 2, j_start - 1]
        ):
            q_out = q_in[0, 2, 0]
        with horizontal(
            region[i_start - 1, j_start - 2], region[i_end + 3, j_start - 1]
        ):
            q_out = q_in[-1, 3, 0]
        with horizontal(
            region[i_start - 1, j_start - 3], region[i_end + 4, j_start - 1]
        ):
            q_out = q_in[-2, 4, 0]
        with horizontal(
            region[i_start - 2, j_start - 1], region[i_end + 2, j_start - 2]
        ):
            q_out = q_in[1, 3, 0]
        with horizontal(
            region[i_start - 2, j_start - 2], region[i_end + 3, j_start - 2]
        ):
            q_out = q_in[0, 4, 0]
        with horizontal(
            region[i_start - 2, j_start - 3], region[i_end + 4, j_start - 2]
        ):
            q_out = q_in[-1, 5, 0]
        with horizontal(
            region[i_start - 3, j_start - 1], region[i_end + 2, j_start - 3]
        ):
            q_out = q_in[2, 4, 0]
        with horizontal(
            region[i_start - 3, j_start - 2], region[i_end + 3, j_start - 3]
        ):
            q_out = q_in[1, 5, 0]
        with horizontal(
            region[i_start - 3, j_start - 3], region[i_end + 4, j_start - 3]
        ):
            q_out = q_in[0, 6, 0]
        # nw and ne corner
        with horizontal(region[i_start - 1, j_end + 2], region[i_end + 2, j_end + 2]):
            q_out = q_in[0, -2, 0]
        with horizontal(region[i_start - 1, j_end + 3], region[i_end + 3, j_end + 2]):
            q_out = q_in[-1, -3, 0]
        with horizontal(region[i_start - 1, j_end + 4], region[i_end + 4, j_end + 2]):
            q_out = q_in[-2, -4, 0]
        with horizontal(region[i_start - 2, j_end + 2], region[i_end + 2, j_end + 3]):
            q_out = q_in[1, -3, 0]
        with horizontal(region[i_start - 2, j_end + 3], region[i_end + 3, j_end + 3]):
            q_out = q_in[0, -4, 0]
        with horizontal(region[i_start - 2, j_end + 4], region[i_end + 4, j_end + 3]):
            q_out = q_in[-1, -5, 0]
        with horizontal(region[i_start - 3, j_end + 2], region[i_end + 2, j_end + 4]):
            q_out = q_in[2, -4, 0]
        with horizontal(region[i_start - 3, j_end + 3], region[i_end + 3, j_end + 4]):
            q_out = q_in[1, -5, 0]
        with horizontal(region[i_start - 3, j_end + 4], region[i_end + 4, j_end + 4]):
            q_out = q_in[0, -6, 0]


def fill_corners_bgrid_y_defn(q_in: FloatField, q_out: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        # sw and nw corners
        with horizontal(
            region[i_start - 1, j_start - 1], region[i_start - 1, j_end + 2]
        ):
            q_out = q_in[2, 0, 0]
        with horizontal(
            region[i_start - 1, j_start - 2], region[i_start - 2, j_end + 2]
        ):
            q_out = q_in[3, 1, 0]
        with horizontal(
            region[i_start - 1, j_start - 3], region[i_start - 3, j_end + 2]
        ):
            q_out = q_in[4, 2, 0]
        with horizontal(
            region[i_start - 2, j_start - 1], region[i_start - 1, j_end + 3]
        ):
            q_out = q_in[3, -1, 0]
        with horizontal(
            region[i_start - 2, j_start - 2], region[i_start - 2, j_end + 3]
        ):
            q_out = q_in[4, 0, 0]
        with horizontal(
            region[i_start - 2, j_start - 3], region[i_start - 3, j_end + 3]
        ):
            q_out = q_in[5, 1, 0]
        with horizontal(
            region[i_start - 3, j_start - 1], region[i_start - 1, j_end + 4]
        ):
            q_out = q_in[4, -2, 0]
        with horizontal(
            region[i_start - 3, j_start - 2], region[i_start - 2, j_end + 4]
        ):
            q_out = q_in[5, -1, 0]
        with horizontal(
            region[i_start - 3, j_start - 3], region[i_start - 3, j_end + 4]
        ):
            q_out = q_in[6, 0, 0]
        # se and ne corners
        with horizontal(region[i_end + 2, j_start - 1], region[i_end + 2, j_end + 2]):
            q_out = q_in[-2, 0, 0]
        with horizontal(region[i_end + 2, j_start - 2], region[i_end + 3, j_end + 2]):
            q_out = q_in[-3, 1, 0]
        with horizontal(region[i_end + 2, j_start - 3], region[i_end + 4, j_end + 2]):
            q_out = q_in[-4, 2, 0]
        with horizontal(region[i_end + 3, j_start - 1], region[i_end + 2, j_end + 3]):
            q_out = q_in[-3, -1, 0]
        with horizontal(region[i_end + 3, j_start - 2], region[i_end + 3, j_end + 3]):
            q_out = q_in[-4, 0, 0]
        with horizontal(region[i_end + 3, j_start - 3], region[i_end + 4, j_end + 3]):
            q_out = q_in[-5, 1, 0]
        with horizontal(region[i_end + 4, j_start - 1], region[i_end + 2, j_end + 4]):
            q_out = q_in[-4, -2, 0]
        with horizontal(region[i_end + 4, j_start - 2], region[i_end + 3, j_end + 4]):
            q_out = q_in[-5, -1, 0]
        with horizontal(region[i_end + 4, j_start - 3], region[i_end + 4, j_end + 4]):
            q_out = q_in[-6, 0, 0]


def fill_sw_corner_2d_agrid(q, i, j, direction, grid, kstart=0, nk=None):
    kslice, nk = utils.kslice_from_inputs(kstart, nk, grid)
    if direction == "x":
        q[grid.is_ - i, grid.js - j, kslice] = q[grid.is_ - j, i, kslice]
    if direction == "y":
        q[grid.is_ - j, grid.js - i, kslice] = q[i, grid.js - j, kslice]


def fill_nw_corner_2d_agrid(q, i, j, direction, grid, kstart=0, nk=None):
    kslice, nk = utils.kslice_from_inputs(kstart, nk, grid)
    if direction == "x":
        q[grid.is_ - i, grid.je + j, kslice] = q[grid.is_ - j, grid.je - i + 1, kslice]
    if direction == "y":
        q[grid.is_ - j, grid.je + i, kslice] = q[i, grid.je + j, kslice]


def fill_se_corner_2d_agrid(q, i, j, direction, grid, kstart=0, nk=None):
    kslice, nk = utils.kslice_from_inputs(kstart, nk, grid)
    if direction == "x":
        q[grid.ie + i, grid.js - j, kslice] = q[grid.ie + j, i, kslice]
    if direction == "y":
        q[grid.ie + j, grid.js - i, kslice] = q[grid.ie - i + 1, grid.js - j, kslice]


def fill_ne_corner_2d_agrid(q, i, j, direction, grid, mysign=1.0, kstart=0, nk=None):
    kslice, nk = utils.kslice_from_inputs(kstart, nk, grid)
    if direction == "x":
        q[grid.ie + i, grid.je + j, kslice] = q[grid.ie + j, grid.je - i + 1, kslice]
    if direction == "y":
        q[grid.ie + j, grid.je + i, kslice] = q[grid.ie - i + 1, grid.je + j, kslice]


def fill_corners_2d_agrid(q, grid, gridtype, direction="x"):
    for i in range(1, 1 + grid.halo):
        for j in range(1, 1 + grid.halo):
            if grid.sw_corner:
                fill_sw_corner_2d_agrid(q, i, j, direction, grid)
            if grid.nw_corner:
                fill_nw_corner_2d_agrid(q, i, j, direction, grid)
            if grid.se_corner:
                fill_se_corner_2d_agrid(q, i, j, direction, grid)
            if grid.ne_corner:
                fill_ne_corner_2d_agrid(q, i, j, direction, grid)


def fill_corners_dgrid_defn(x: FloatField, y: FloatField, mysign: float):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        # sw corner
        with horizontal(region[i_start - 1, j_start - 1]):
            x = mysign * y[0, 1, 0]
        with horizontal(region[i_start - 1, j_start - 1]):
            y = mysign * x[1, 0, 0]
        with horizontal(region[i_start - 1, j_start - 2]):
            x = mysign * y[-1, 2, 0]
        with horizontal(region[i_start - 1, j_start - 2]):
            y = mysign * x[2, 1, 0]
        with horizontal(region[i_start - 1, j_start - 3]):
            x = mysign * y[-2, 3, 0]
        with horizontal(region[i_start - 1, j_start - 3]):
            y = mysign * x[3, 2, 0]
        with horizontal(region[i_start - 2, j_start - 1]):
            x = mysign * y[1, 2, 0]
        with horizontal(region[i_start - 2, j_start - 1]):
            y = mysign * x[2, -1, 0]
        with horizontal(region[i_start - 2, j_start - 2]):
            x = mysign * y[0, 3, 0]
        with horizontal(region[i_start - 2, j_start - 2]):
            y = mysign * x[3, 0, 0]
        with horizontal(region[i_start - 2, j_start - 3]):
            x = mysign * y[-1, 4, 0]
        with horizontal(region[i_start - 2, j_start - 3]):
            y = mysign * x[4, 1, 0]
        with horizontal(region[i_start - 3, j_start - 1]):
            x = mysign * y[2, 3, 0]
        with horizontal(region[i_start - 3, j_start - 1]):
            y = mysign * x[3, -2, 0]
        with horizontal(region[i_start - 3, j_start - 2]):
            x = mysign * y[1, 4, 0]
        with horizontal(region[i_start - 3, j_start - 2]):
            y = mysign * x[4, -1, 0]
        with horizontal(region[i_start - 3, j_start - 3]):
            x = mysign * y[0, 5, 0]
        with horizontal(region[i_start - 3, j_start - 3]):
            y = mysign * x[5, 0, 0]
        # ne corner
        with horizontal(region[i_end + 1, j_end + 2]):
            x = mysign * y[1, -2, 0]
        with horizontal(region[i_end + 2, j_end + 1]):
            y = mysign * x[-2, 1, 0]
        with horizontal(region[i_end + 1, j_end + 3]):
            x = mysign * y[2, -3, 0]
        with horizontal(region[i_end + 2, j_end + 2]):
            y = mysign * x[-3, 0, 0]
        with horizontal(region[i_end + 1, j_end + 4]):
            x = mysign * y[3, -4, 0]
        with horizontal(region[i_end + 2, j_end + 3]):
            y = mysign * x[-4, -1, 0]
        with horizontal(region[i_end + 2, j_end + 2]):
            x = mysign * y[0, -3, 0]
        with horizontal(region[i_end + 3, j_end + 1]):
            y = mysign * x[-3, 2, 0]
        with horizontal(region[i_end + 2, j_end + 3]):
            x = mysign * y[1, -4, 0]
        with horizontal(region[i_end + 3, j_end + 2]):
            y = mysign * x[-4, 1, 0]
        with horizontal(region[i_end + 2, j_end + 4]):
            x = mysign * y[2, -5, 0]
        with horizontal(region[i_end + 3, j_end + 3]):
            y = mysign * x[-5, 0, 0]
        with horizontal(region[i_end + 3, j_end + 2]):
            x = mysign * y[-1, -4, 0]
        with horizontal(region[i_end + 4, j_end + 1]):
            y = mysign * x[-4, 3, 0]
        with horizontal(region[i_end + 3, j_end + 3]):
            x = mysign * y[0, -5, 0]
        with horizontal(region[i_end + 4, j_end + 2]):
            y = mysign * x[-5, 2, 0]
        with horizontal(region[i_end + 3, j_end + 4]):
            x = mysign * y[1, -6, 0]
        with horizontal(region[i_end + 4, j_end + 3]):
            y = mysign * x[-6, 1, 0]
        # nw corner
        with horizontal(region[i_start - 1, j_end + 2]):
            x = y[0, -2, 0]
        with horizontal(region[i_start - 1, j_end + 1]):
            y = x[1, 1, 0]
        with horizontal(region[i_start - 1, j_end + 3]):
            x = y[-1, -3, 0]
        with horizontal(region[i_start - 1, j_end + 2]):
            y = x[2, 0, 0]
        with horizontal(region[i_start - 1, j_end + 4]):
            x = y[-2, -4, 0]
        with horizontal(region[i_start - 1, j_end + 3]):
            y = x[3, -1, 0]
        with horizontal(region[i_start - 2, j_end + 2]):
            x = y[1, -3, 0]
        with horizontal(region[i_start - 2, j_end + 1]):
            y = x[2, 2, 0]
        with horizontal(region[i_start - 2, j_end + 3]):
            x = y[0, -4, 0]
        with horizontal(region[i_start - 2, j_end + 2]):
            y = x[3, 1, 0]
        with horizontal(region[i_start - 2, j_end + 4]):
            x = y[-1, -5, 0]
        with horizontal(region[i_start - 2, j_end + 3]):
            y = x[4, 0, 0]
        with horizontal(region[i_start - 3, j_end + 2]):
            x = y[2, -4, 0]
        with horizontal(region[i_start - 3, j_end + 1]):
            y = x[3, 3, 0]
        with horizontal(region[i_start - 3, j_end + 3]):
            x = y[1, -5, 0]
        with horizontal(region[i_start - 3, j_end + 2]):
            y = x[4, 2, 0]
        with horizontal(region[i_start - 3, j_end + 4]):
            x = y[0, -6, 0]
        with horizontal(region[i_start - 3, j_end + 3]):
            y = x[5, 1, 0]
        # se corner
        with horizontal(region[i_end + 1, j_start - 1]):
            x = y[1, 1, 0]
        with horizontal(region[i_end + 2, j_start - 1]):
            y = x[-2, 0, 0]
        with horizontal(region[i_end + 1, j_start - 2]):
            x = y[2, 2, 0]
        with horizontal(region[i_end + 2, j_start - 2]):
            y = x[-3, 1, 0]
        with horizontal(region[i_end + 1, j_start - 3]):
            x = y[3, 3, 0]
        with horizontal(region[i_end + 2, j_start - 3]):
            y = x[-4, 2, 0]
        with horizontal(region[i_end + 2, j_start - 1]):
            x = y[0, 2, 0]
        with horizontal(region[i_end + 3, j_start - 1]):
            y = x[-3, -1, 0]
        with horizontal(region[i_end + 2, j_start - 2]):
            x = y[1, 3, 0]
        with horizontal(region[i_end + 3, j_start - 2]):
            y = x[-4, 0, 0]
        with horizontal(region[i_end + 2, j_start - 3]):
            x = y[2, 4, 0]
        with horizontal(region[i_end + 3, j_start - 3]):
            y = x[-5, 1, 0]
        with horizontal(region[i_end + 3, j_start - 1]):
            x = y[-1, 3, 0]
        with horizontal(region[i_end + 4, j_start - 1]):
            y = x[-4, -2, 0]
        with horizontal(region[i_end + 3, j_start - 2]):
            x = y[0, 4, 0]
        with horizontal(region[i_end + 4, j_start - 2]):
            y = x[-5, -1, 0]
        with horizontal(region[i_end + 3, j_start - 3]):
            x = y[1, 5, 0]
        with horizontal(region[i_end + 4, j_start - 3]):
            y = x[-6, 0, 0]


@gtscript.function
def corner_ke(
    ke,
    u,
    v,
    ut,
    vt,
    dt,
    io1,
    jo1,
    io2,
    vsign,
):
    dt6 = dt / 6.0

    ke = dt6 * (
        (ut[0, 0, 0] + ut[0, -1, 0]) * ((io1 + 1) * u[0, 0, 0] - (io1 * u[-1, 0, 0]))
        + (vt[0, 0, 0] + vt[-1, 0, 0]) * ((jo1 + 1) * v[0, 0, 0] - (jo1 * v[0, -1, 0]))
        + (
            ((jo1 + 1) * ut[0, 0, 0] - (jo1 * ut[0, -1, 0]))
            + vsign * ((io1 + 1) * vt[0, 0, 0] - (io1 * vt[-1, 0, 0]))
        )
        * ((io2 + 1) * u[0, 0, 0] - (io2 * u[-1, 0, 0]))
    )

    return ke
