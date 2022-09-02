from gt4py.gtscript import BACKWARD, PARALLEL, computation, interval

from pace.dsl.typing import FloatField
from pace.util.constants import ZVIR


def get_phi_fv3(
    gt0: FloatField,
    gq0: FloatField,
    del_gz: FloatField,
    phii: FloatField,
    phil: FloatField,
):
    with computation(PARALLEL), interval(0, -1):
        del_gz = del_gz[0, 0, 0] * gt0[0, 0, 0] * (1.0 + ZVIR * max(0.0, gq0[0, 0, 0]))

    with computation(BACKWARD):
        with interval(-1, None):
            phii = 0.0
        with interval(-2, -1):
            phil = 0.5 * (phii[0, 0, 1] + phii[0, 0, 1] + del_gz[0, 0, 0])
            phii = phii[0, 0, 1] + del_gz[0, 0, 0]
        with interval(0, -2):
            phil = 0.5 * (phii[0, 0, 1] + phii[0, 0, 1] + del_gz[0, 0, 0])
            phii = phii[0, 0, 1] + del_gz[0, 0, 0]
