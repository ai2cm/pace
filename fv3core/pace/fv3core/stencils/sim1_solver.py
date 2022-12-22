import typing

from gt4py.cartesian.gtscript import (
    BACKWARD,
    FORWARD,
    PARALLEL,
    computation,
    exp,
    interval,
    log,
)

import pace.util.constants as constants
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ
from pace.util import X_DIM, Y_DIM, Z_INTERFACE_DIM


@typing.no_type_check
def sim1_solver(
    w: FloatField,
    delta_mass: FloatField,
    gamma: FloatField,
    dz: FloatField,
    potential_temperature: FloatField,
    pm: FloatField,
    pe: FloatField,
    pem: FloatField,
    ws: FloatFieldIJ,
    cp3: FloatField,
    dt: float,
    t1g: float,
    rdt: float,
    p_fac: float,
):
    """
    Tridiagonal solve for w and dz, handles pressure gradient force and sound waves
    in the vertical.

    Documented in Chapter 7.1 of the FV3 dynamical core documentation.

    Args:
        w (inout):
        delta_mass (in):
        gamma (in):
        dz (inout):
        ptr (in):
        pm (in):
        pe (out): nonhydrostatic perturbation pressure defined on interfaces
        pem (in):
        ws (in): surface vertical wind (e.g. due to topography)
        cp3 (in):
    """
    with computation(PARALLEL), interval(0, -1):
        pe = (
            exp(gamma * log(-delta_mass / dz * constants.RDGAS * potential_temperature))
            - pm
        )
        w1 = w
    with computation(FORWARD):
        with interval(0, -2):
            g_rat = delta_mass / delta_mass[0, 0, 1]
            bb = 2.0 * (1.0 + g_rat)
            dd = 3.0 * (pe + g_rat * pe[0, 0, 1])
        with interval(-2, -1):
            bb = 2.0
            dd = 3.0 * pe
    # bet[i,j,k] = bb[i,j,0]
    with computation(FORWARD):
        with interval(0, 1):
            bet = bb
        with interval(1, -1):
            bet = bet[0, 0, -1]

    # stencils: w_solver
    # {
    with computation(PARALLEL):
        with interval(0, 1):
            pp = 0.0
        with interval(1, 2):
            pp = dd[0, 0, -1] / bet
    with computation(FORWARD), interval(1, -1):
        gam = g_rat[0, 0, -1] / bet[0, 0, -1]
        bet = bb - gam
    with computation(FORWARD), interval(2, None):
        pp = (dd[0, 0, -1] - pp[0, 0, -1]) / bet[0, 0, -1]
    with computation(BACKWARD), interval(1, -1):
        pp = pp - gam * pp[0, 0, 1]
        # w solver
        aa = t1g * 0.5 * (gamma[0, 0, -1] + gamma) / (dz[0, 0, -1] + dz) * (pem + pp)
    # }
    # updates on bet:
    with computation(FORWARD):
        with interval(0, 1):
            bet = delta_mass[0, 0, 0] - aa[0, 0, 1]
        with interval(1, None):
            bet = bet[0, 0, -1]
    # w_pe_dz_compute
    # {
    with computation(FORWARD):
        with interval(0, 1):
            w = (delta_mass * w1 + dt * pp[0, 0, 1]) / bet
        with interval(1, -2):
            gam = aa / bet[0, 0, -1]
            bet = delta_mass - (aa + aa[0, 0, 1] + aa * gam)
            w = (delta_mass * w1 + dt * (pp[0, 0, 1] - pp) - aa * w[0, 0, -1]) / bet
        with interval(-2, -1):
            p1 = t1g * gamma / dz * (pem[0, 0, 1] + pp[0, 0, 1])
            gam = aa / bet[0, 0, -1]
            bet = delta_mass - (aa + p1 + aa * gam)
            w = (
                delta_mass * w1
                + dt * (pp[0, 0, 1] - pp)
                - p1 * ws[0, 0]
                - aa * w[0, 0, -1]
            ) / bet
    with computation(BACKWARD), interval(0, -2):
        w = w - gam[0, 0, 1] * w[0, 0, 1]
    with computation(FORWARD):
        with interval(0, 1):
            pe = 0.0
        with interval(1, None):
            pe = (
                pe[0, 0, -1] + delta_mass[0, 0, -1] * (w[0, 0, -1] - w1[0, 0, -1]) * rdt
            )
    with computation(BACKWARD):
        with interval(-2, -1):
            p1 = (pe + 2.0 * pe[0, 0, 1]) * 1.0 / 3.0
        with interval(0, -2):
            p1 = (pe + bb * pe[0, 0, 1] + g_rat * pe[0, 0, 2]) * 1.0 / 3.0 - g_rat * p1[
                0, 0, 1
            ]
    with computation(PARALLEL), interval(0, -1):
        maxp = p_fac * pm if p_fac * delta_mass > p1 + pm else p1 + pm
        dz = (
            -delta_mass
            * constants.RDGAS
            * potential_temperature
            * exp((cp3 - 1.0) * log(maxp))
        )
    # }


class Sim1Solver:
    """
    Fortran name is sim1_solver

    Namelist:
        p_fac: Safety factor for minimum nonhydrostatic pressures.
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        p_fac: float,
        n_halo: int,
    ):
        self._pfac = p_fac
        self._compute_sim1_solve = stencil_factory.from_dims_halo(
            func=sim1_solver,
            compute_dims=[X_DIM, Y_DIM, Z_INTERFACE_DIM],
            compute_halos=(n_halo, n_halo, 0),
        )

    def __call__(
        self,
        dt: float,
        gamma: FloatField,
        cp3: FloatField,
        pe: FloatField,
        delta_mass: FloatField,
        pm: FloatField,
        pem: FloatField,
        w: FloatField,
        dz: FloatField,
        potential_temperature: FloatField,
        ws: FloatFieldIJ,
    ):
        """
        Semi-Implicit Method solver -- solves a vertically tridiagonal
        system for sound waves to compute nonhydrostatic terms for
        vertical velocity and pressure perturbations.

        Chapter 7 of the FV3 documentation

        Args:
          dt (in): timstep in seconds of solver
          gm (in): ?? 1 / (1 - cappa)
          cp3 (in): cappa
          pe (out): full hydrostatic pressure
          delta_mass (in): mass thickness of atmospheric layer (kg)
          pm (in): ?? ratio of change in layer pressure without condensates
          pem (in): recomputed pressure using ptop and delp
          w (inout): vertical velocity
          dz (inout): vertical delta of atmospheric layer in meters
          potential_temperature (in): potential temperature
          ws (in): surface vertical wind (e.g. due to topography)
        """

        # TODO: email Lucas about any remaining variable naming here

        t1g = 2.0 * dt * dt
        rdt = 1.0 / dt
        self._compute_sim1_solve(
            w,
            delta_mass,
            gamma,
            dz,
            potential_temperature,
            pm,
            pe,
            pem,
            ws,
            cp3,
            dt,
            t1g,
            rdt,
            self._pfac,
        )
