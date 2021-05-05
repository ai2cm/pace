from typing import Dict, Tuple

from gt4py.gtscript import FORWARD, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil, gtstencil
from fv3core.stencils.basic_operations import copy_defn
from fv3core.stencils.remap_profile import RemapProfile
from fv3core.utils.typing import FloatField, FloatFieldIJ


def set_dp(dp1: FloatField, pe1: FloatField):
    with computation(PARALLEL), interval(...):
        dp1 = pe1[0, 0, 1] - pe1


def set_eulerian_pressures(pe: FloatField, ptop: FloatFieldIJ, pbot: FloatFieldIJ):
    with computation(FORWARD), interval(0, 1):
        ptop = pe[0, 0, 0]
        pbot = pe[0, 0, 1]


def set_remapped_quantity(q: FloatField, set_values: FloatFieldIJ):
    with computation(FORWARD), interval(0, 1):
        q = set_values[0, 0]


def lagrangian_contributions(
    pe1: FloatField,
    ptop: FloatFieldIJ,
    pbot: FloatFieldIJ,
    q4_1: FloatField,
    q4_2: FloatField,
    q4_3: FloatField,
    q4_4: FloatField,
    dp1: FloatField,
    q2_adds: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        q2_tmp = 0.0
        if pe1 < pbot and pe1[0, 0, 1] > ptop:
            # We are in the right pressure range to contribute to the Eulerian cell
            if pe1 <= ptop:
                # we are in the first Lagrangian level that conributes
                pl = (ptop - pe1) / dp1
                if pbot <= pe1[0, 0, 1]:
                    # Eulerian grid element is contained in the Lagrangian element
                    pr = (pbot - pe1) / dp1
                    q2_tmp = (
                        q4_2
                        + 0.5 * (q4_4 + q4_3 - q4_2) * (pr + pl)
                        - q4_4 * (1.0 / 3.0) * (pr * (pr + pl) + pl ** 2)
                    )
                else:
                    # Eulerian element encompasses multiple Lagrangian elements
                    # and this is just the first one
                    q2_tmp = (
                        (pe1[0, 0, 1] - ptop)
                        * (
                            q4_2
                            + 0.5 * (q4_4 + q4_3 - q4_2) * (1.0 + pl)
                            - q4_4 * (1.0 / 3.0) * (1.0 + pl * (1.0 + pl))
                        )
                        / (pbot - ptop)
                    )
            else:
                # we are in a farther-down level
                if pbot > pe1[0, 0, 1]:
                    # add the whole level to the Eulerian cell
                    q2_tmp = dp1 * q4_1 / (pbot - ptop)
                else:
                    # this is the bottom layer that contributes
                    dp = pbot - pe1
                    esl = dp / dp1
                    q2_tmp = (
                        dp
                        * (
                            q4_2
                            + 0.5
                            * esl
                            * (q4_3 - q4_2 + q4_4 * (1.0 - (2.0 / 3.0) * esl))
                        )
                        / (pbot - ptop)
                    )
    with computation(FORWARD), interval(0, 1):
        q2_adds = 0.0
    with computation(FORWARD), interval(...):
        q2_adds += q2_tmp


class LagrangianContributions:
    """
    Stencil implementation of remap_z, map1_ppm, map_scalar, in Fortran
    """

    def __init__(self, origin: Tuple[int, int, int], domain: Tuple[int, int, int]):
        self._grid = spec.grid
        self._km = self._grid.npz
        shape = self._grid.domain_shape_full(add=(1, 1, 1))
        shape2d = shape[0:2]

        self._q2_adds = utils.make_storage_from_shape(shape2d)
        self._ptop = utils.make_storage_from_shape(shape2d)
        self._pbot = utils.make_storage_from_shape(shape2d)

        self.origin = origin
        self.domain = domain

        self._set_eulerian_pressures = gtstencil(set_eulerian_pressures)
        self._lagrangian_contributions = FrozenStencil(
            lagrangian_contributions,
            origin=origin,
            domain=domain,
        )
        self._set_remapped_quantity = gtstencil(set_remapped_quantity)

    def __call__(
        self,
        q1: FloatField,
        pe1: FloatField,
        pe2: FloatField,
        q4_1: FloatField,
        q4_2: FloatField,
        q4_3: FloatField,
        q4_4: FloatField,
        dp1: FloatField,
    ):
        """
        Distributes a field from the deformed Lagrangian grid onto the remapped
        Eulerian grid.

        Args:
            q1 (out): the field on the remapped grid
            pe1 (in): Lagrangian pressure levels
            pe2 (in): Eulerian pressure levels
            q4_1, q4_2, q4_3, a4_4 (in): the interpolation coefficients that specify
                the cubic subgrid distribution of q on the deformed grid.
            dp1 (in): the pressure difference between the top and bottom of each
                Lagrangian level
        """
        eul_domain = (self.domain[0], self.domain[1], 1)

        # A stencil with a loop over k2:
        for k_eul in range(self._km):
            eul_origin = (self.origin[0], self.origin[1], k_eul)

            # TODO (olivere): This is hacky
            # merge with subsequent stencil when possible
            self._set_eulerian_pressures(
                pe2,
                self._ptop,
                self._pbot,
                origin=eul_origin,
                domain=eul_domain,
            )

            self._lagrangian_contributions(
                pe1,
                self._ptop,
                self._pbot,
                q4_1,
                q4_2,
                q4_3,
                q4_4,
                dp1,
                self._q2_adds,
            )

            self._set_remapped_quantity(
                q1,
                self._q2_adds,
                origin=eul_origin,
                domain=eul_domain,
            )


class MapSingle:
    """
    Fortran name is map_single, test classes are Map1_PPM_2d, Map_Scalar_2d
    """

    def __init__(self, kord: int, mode: int, i1: int, i2: int, j1: int, j2: int):
        self._grid = spec.grid
        shape = self._grid.domain_shape_full(add=(1, 1, 1))
        origin = self._grid.compute_origin()

        self.dp1 = utils.make_storage_from_shape(shape, origin=origin)
        self.q4_1 = utils.make_storage_from_shape(shape, origin=origin)
        self.q4_2 = utils.make_storage_from_shape(shape, origin=origin)
        self.q4_3 = utils.make_storage_from_shape(shape, origin=origin)
        self.q4_4 = utils.make_storage_from_shape(shape, origin=origin)

        self.i_extent = i2 - i1 + 1
        self.j_extent = j2 - j1 + 1
        origin = (i1, j1, 0)
        domain = (self.i_extent, self.j_extent, self._grid.npz)

        self._lagrangian_contributions = LagrangianContributions(origin, domain)
        self._remap_profile = RemapProfile(kord, mode, i1, i2, j1, j2)
        self._set_dp = FrozenStencil(set_dp, origin=origin, domain=domain)
        self._copy_stencil = FrozenStencil(
            copy_defn,
            origin=(0, 0, 0),
            domain=self._grid.domain_shape_full(),
        )

    def __call__(
        self,
        q1: FloatField,
        pe1: FloatField,
        pe2: FloatField,
        qs: FloatField,
        qmin: float = 0.0,
    ):
        """
        Compute x-flux using the PPM method.

        Args:
            q1 (out): Remapped field on Eulerian grid
            pe1 (in): Lagrangian pressure levels
            pe2 (in): Eulerian pressure levels
            qs (in): Field to be remapped on deformed grid
            jfirst: Starting index of the J-dir compute domain
            jlast: Final index of the J-dir compute domain
        """
        self._copy_stencil(q1, self.q4_1)
        self._set_dp(self.dp1, pe1)
        q4_1, q4_2, q4_3, q4_4 = self._remap_profile(
            qs,
            self.q4_1,
            self.q4_2,
            self.q4_3,
            self.q4_4,
            self.dp1,
            qmin,
        )
        self._lagrangian_contributions(
            q1,
            pe1,
            pe2,
            q4_1,
            q4_2,
            q4_3,
            q4_4,
            self.dp1,
        )
        return q1


class MapSingleFactory:
    _object_pool: Dict[Tuple[int, ...], MapSingle] = {}
    """Pool of MapSingle objects."""

    def __call__(
        self, kord: int, mode: int, i1: int, i2: int, j1: int, j2: int, *args, **kwargs
    ):
        key_tuple = (kord, mode, i1, i2, j1, j2)
        if key_tuple not in self._object_pool:
            self._object_pool[key_tuple] = MapSingle(*key_tuple)
        return self._object_pool[key_tuple](*args, **kwargs)
