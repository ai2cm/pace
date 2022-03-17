from gt4py.gtscript import PARALLEL, computation, interval

from pace import util
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField


def reset_tendencies(u_dt: FloatField, v_dt: FloatField, pt_dt: FloatField):
    with computation(PARALLEL), interval(...):
        u_dt = 0.0
        v_dt = 0.0
        pt_dt = 0.0


class ResetTendencies:
    """
    A class for setting tendencies to 0, since slice assignment
    for 3d gpu storages doesn't work
    """

    def __init__(self, stencil_factory: StencilFactory):

        self._reset_tendencies = stencil_factory.from_dims_halo(
            func=reset_tendencies,
            compute_dims=[
                util.X_INTERFACE_DIM,
                util.Y_INTERFACE_DIM,
                util.Z_INTERFACE_DIM,
            ],
            compute_halos=(0, 0),
        )

    def __call__(self, u_dt: util.Quantity, v_dt: util.Quantity, pt_dt: util.Quantity):
        """
        Zeros out tendencies
        u_dt: x-wind tendency (out)
        v_dt: y-wind tendency (out)
        pt_dt: temperature tendency (out)

        """
        self._reset_tendencies(u_dt.storage, v_dt.storage, pt_dt.storage)
