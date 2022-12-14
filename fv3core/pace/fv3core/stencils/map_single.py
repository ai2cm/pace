from typing import Optional, Sequence

from gt4py.cartesian.gtscript import FORWARD, PARALLEL, computation, interval

import pace.util
from pace.dsl.dace import orchestrate
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ, IntFieldIJ  # noqa: F401
from pace.fv3core.stencils.basic_operations import copy_defn
from pace.fv3core.stencils.remap_profile import RemapProfile
from pace.util import X_DIM, Y_DIM, Z_DIM


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
    """
    Args:
        q (out):
        pe1 (in):
        pe2 (in):
        q4_1 (in):
        q4_2 (in):
        q4_3 (in):
        q4_4 (in):
        dp1 (in):
        lev (inout):
    """
    # TODO: Can we make lev a 2D temporary?
    with computation(FORWARD), interval(...):
        pl = (pe2 - pe1[0, 0, lev]) / dp1[0, 0, lev]
        if pe2[0, 0, 1] <= pe1[0, 0, lev + 1]:
            pr = (pe2[0, 0, 1] - pe1[0, 0, lev]) / dp1[0, 0, lev]
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

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: pace.util.QuantityFactory,
        kord: int,
        mode: int,
        dims: Sequence[str],
    ):
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
        )

        # TODO: consider refactoring to take in origin and domain
        grid_indexing = stencil_factory.grid_indexing

        def make_quantity():
            return quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], units="unknown")

        self._dp1 = make_quantity()
        self._q4_1 = make_quantity()
        self._q4_2 = make_quantity()
        self._q4_3 = make_quantity()
        self._q4_4 = make_quantity()
        self._tmp_qs = quantity_factory.zeros([X_DIM, Y_DIM], units="unknown")
        self._lev = quantity_factory.zeros([X_DIM, Y_DIM], units="", dtype=int)

        self._copy_stencil = stencil_factory.from_dims_halo(
            copy_defn,
            compute_dims=dims,
        )

        self._set_dp = stencil_factory.from_dims_halo(
            set_dp,
            compute_dims=dims,
        )

        self._remap_profile = RemapProfile(
            stencil_factory,
            quantity_factory,
            kord,
            mode,
            dims=dims,
        )

        self._lagrangian_contributions = stencil_factory.from_dims_halo(
            lagrangian_contributions,
            compute_dims=dims,
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
            qmin (in): Minimum allowed value of the remapped field
        """

        self._copy_stencil(q1, self._q4_1)
        self._set_dp(self._dp1, pe1, self._lev)

        if qs is None:
            self._remap_profile(
                self._tmp_qs,
                self._q4_1,
                self._q4_2,
                self._q4_3,
                self._q4_4,
                self._dp1,
                qmin,
            )
        else:
            self._remap_profile(
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
            self._q4_1,
            self._q4_2,
            self._q4_3,
            self._q4_4,
            self._dp1,
            self._lev,
        )
        return q1
