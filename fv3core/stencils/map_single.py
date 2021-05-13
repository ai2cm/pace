from typing import Dict, Optional, Tuple

from gt4py.gtscript import FORWARD, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil
from fv3core.stencils.basic_operations import copy_defn
from fv3core.stencils.remap_profile import RemapProfile
from fv3core.utils.typing import FloatField, FloatFieldIJ, IntFieldIJ


def set_dp(dp1: FloatField, pe1: FloatField, lev: IntFieldIJ):
    with computation(PARALLEL), interval(...):
        dp1 = pe1[0, 0, 1] - pe1
    with computation(FORWARD), interval(0, 1):
        lev = 0


def lagrangian_contributions(
    q: FloatField,
    pe1: FloatField,
    pe2: FloatField,
    q4_1: FloatField,
    q4_2: FloatField,
    q4_3: FloatField,
    q4_4: FloatField,
    dp1: FloatField,
    lev: IntFieldIJ,
):
    with computation(FORWARD), interval(...):
        v_pe2 = pe2
        v_pe1 = pe1[0, 0, lev]
        pl = (v_pe2 - v_pe1) / dp1[0, 0, lev]
        if pe2[0, 0, 1] <= pe1[0, 0, lev + 1]:
            pr = (pe2[0, 0, 1] - v_pe1) / dp1[0, 0, lev]
            q = (
                q4_2[0, 0, lev]
                + 0.5
                * (q4_4[0, 0, lev] + q4_3[0, 0, lev] - q4_2[0, 0, lev])
                * (pr + pl)
                - q4_4[0, 0, lev] * 1.0 / 3.0 * (pr * (pr + pl) + pl * pl)
            )
        else:
            qsum = (pe1[0, 0, lev + 1] - pe2) * (
                q4_2[0, 0, lev]
                + 0.5
                * (q4_4[0, 0, lev] + q4_3[0, 0, lev] - q4_2[0, 0, lev])
                * (1.0 + pl)
                - q4_4[0, 0, lev] * 1.0 / 3.0 * (1.0 + pl * (1.0 + pl))
            )
            lev = lev + 1
            while pe1[0, 0, lev + 1] < pe2[0, 0, 1]:
                qsum += dp1[0, 0, lev] * q4_1[0, 0, lev]
                lev = lev + 1
            dp = pe2[0, 0, 1] - pe1[0, 0, lev]
            esl = dp / dp1[0, 0, lev]
            qsum += dp * (
                q4_2[0, 0, lev]
                + 0.5
                * esl
                * (
                    q4_3[0, 0, lev]
                    - q4_2[0, 0, lev]
                    + q4_4[0, 0, lev] * (1.0 - (2.0 / 3.0) * esl)
                )
            )
            q = qsum / (pe2[0, 0, 1] - pe2)
        lev = lev - 1


class MapSingle:
    """
    Fortran name is map_single, test classes are Map1_PPM_2d, Map_Scalar_2d
    """

    def __init__(self, kord: int, mode: int, i1: int, i2: int, j1: int, j2: int):
        grid = spec.grid
        shape = grid.domain_shape_full(add=(1, 1, 1))
        origin = grid.compute_origin()

        self._dp1 = utils.make_storage_from_shape(shape, origin=origin)
        self._q4_1 = utils.make_storage_from_shape(shape, origin=origin)
        self._q4_2 = utils.make_storage_from_shape(shape, origin=origin)
        self._q4_3 = utils.make_storage_from_shape(shape, origin=origin)
        self._q4_4 = utils.make_storage_from_shape(shape, origin=origin)
        self._tmp_qs = utils.make_storage_from_shape(shape[0:2], origin=(0, 0))
        self._lev = utils.make_storage_from_shape(
            shape[:-1],
            origin=origin[:-1],
            mask=(True, True, False),
            dtype=int,
        )

        self._extents = (i2 - i1 + 1, j2 - j1 + 1)
        origin = (i1, j1, 0)
        domain = (*self._extents, grid.npz)

        self._lagrangian_contributions = FrozenStencil(
            lagrangian_contributions,
            origin=origin,
            domain=domain,
        )
        self._remap_profile = RemapProfile(kord, mode, i1, i2, j1, j2)

        self._set_dp = FrozenStencil(set_dp, origin=origin, domain=domain)
        self._copy_stencil = FrozenStencil(
            copy_defn,
            origin=(0, 0, 0),
            domain=grid.domain_shape_full(),
        )

    @property
    def i_extent(self):
        return self._extents[0]

    @property
    def j_extent(self):
        return self._extents[1]

    def __call__(
        self,
        q1: FloatField,
        pe1: FloatField,
        pe2: FloatField,
        qs: Optional["FloatFieldIJ"] = None,
        qmin: float = 0.0,
    ):
        """
        Compute x-flux using the PPM method.

        Args:
            q1 (out): Remapped field on Eulerian grid
            pe1 (in): Lagrangian pressure levels
            pe2 (in): Eulerian pressure levels
            qs (in): Bottom boundary condition
            jfirst: Starting index of the J-dir compute domain
            jlast: Final index of the J-dir compute domain
        """
        if qs is None:
            qs = self._tmp_qs
        self._copy_stencil(q1, self._q4_1)
        self._set_dp(self._dp1, pe1, self._lev)
        q4_1, q4_2, q4_3, q4_4 = self._remap_profile(
            qs,
            self._q4_1,
            self._q4_2,
            self._q4_3,
            self._q4_4,
            self._dp1,
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
            self._dp1,
            self._lev,
        )
        return q1


# TODO: move this class to the testing code, it is only used there
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
