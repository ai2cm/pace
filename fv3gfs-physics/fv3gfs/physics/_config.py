import dataclasses
from typing import Tuple

import f90nml

from pace.util.namelist import NamelistDefaults, namelist_to_flatish_dict


@dataclasses.dataclass
class PhysicsConfig:
    dt_atmos: int
    hydrostatic: bool
    npx: int
    npy: int
    npz: int
    nwat: int
    do_qa: bool
    c_cracw: float = NamelistDefaults.c_cracw
    c_paut: float = NamelistDefaults.c_paut
    c_pgacs: float = NamelistDefaults.c_pgacs
    c_psaci: float = NamelistDefaults.c_psaci
    ccn_l: float = NamelistDefaults.ccn_l
    ccn_o: float = NamelistDefaults.ccn_o
    const_vg: bool = NamelistDefaults.const_vg
    const_vi: bool = NamelistDefaults.const_vi
    const_vr: bool = NamelistDefaults.const_vr
    const_vs: bool = NamelistDefaults.const_vs
    vs_fac: float = NamelistDefaults.vs_fac
    vg_fac: float = NamelistDefaults.vg_fac
    vi_fac: float = NamelistDefaults.vi_fac
    vr_fac: float = NamelistDefaults.vr_fac
    de_ice: bool = NamelistDefaults.de_ice
    layout: Tuple[int, int] = NamelistDefaults.layout
    # gfdl_cloud_microphys.F90
    tau_imlt: float = NamelistDefaults.tau_imlt  # cloud ice melting
    tau_i2s: float = NamelistDefaults.tau_i2s  # cloud ice to snow auto - conversion
    tau_g2v: float = NamelistDefaults.tau_g2v  # graupel sublimation
    tau_v2g: float = (
        NamelistDefaults.tau_v2g
    )  # graupel deposition -- make it a slow process
    ql_mlt: float = (
        NamelistDefaults.ql_mlt
    )  # max value of cloud water allowed from melted cloud ice
    qs_mlt: float = NamelistDefaults.qs_mlt  # max cloud water due to snow melt
    t_sub: float = NamelistDefaults.t_sub  # min temp for sublimation of cloud ice
    qi_gen: float = (
        NamelistDefaults.qi_gen
    )  # max cloud ice generation during remapping step
    qi_lim: float = (
        NamelistDefaults.qi_lim
    )  # cloud ice limiter to prevent large ice build up
    qi0_max: float = NamelistDefaults.qi0_max  # max cloud ice value (by other sources)
    rad_snow: bool = (
        NamelistDefaults.rad_snow
    )  # consider snow in cloud fraction calculation
    rad_rain: bool = (
        NamelistDefaults.rad_rain
    )  # consider rain in cloud fraction calculation
    dw_ocean: float = NamelistDefaults.dw_ocean  # base value for ocean
    dw_land: float = (
        NamelistDefaults.dw_land
    )  # base value for subgrid deviation / variability over land
    # cloud scheme 0 - ?
    # 1: old fvgfs gfdl) mp implementation
    # 2: binary cloud scheme (0 / 1)
    tau_l2v: float = (
        NamelistDefaults.tau_l2v
    )  # cloud water to water vapor (evaporation)
    c2l_ord: int = NamelistDefaults.c2l_ord
    do_sedi_heat: bool = NamelistDefaults.do_sedi_heat
    do_sedi_w: bool = NamelistDefaults.do_sedi_w
    fast_sat_adj: bool = NamelistDefaults.fast_sat_adj
    qc_crt: float = NamelistDefaults.qc_crt
    fix_negative: bool = NamelistDefaults.fix_negative
    irain_f: int = NamelistDefaults.irain_f
    mp_time: float = NamelistDefaults.mp_time
    prog_ccn: bool = NamelistDefaults.prog_ccn
    qi0_crt: float = NamelistDefaults.qi0_crt
    qs0_crt: float = NamelistDefaults.qs0_crt
    rh_inc: float = NamelistDefaults.rh_inc
    rh_inr: float = NamelistDefaults.rh_inr
    # rh_ins: Any
    rthresh: float = NamelistDefaults.rthresh
    sedi_transport: bool = NamelistDefaults.sedi_transport
    # use_ccn: Any
    use_ppm: bool = NamelistDefaults.use_ppm
    vg_max: float = NamelistDefaults.vg_max
    vi_max: float = NamelistDefaults.vi_max
    vr_max: float = NamelistDefaults.vr_max
    vs_max: float = NamelistDefaults.vs_max
    z_slope_ice: bool = NamelistDefaults.z_slope_ice
    z_slope_liq: bool = NamelistDefaults.z_slope_liq
    tice: float = NamelistDefaults.tice
    alin: float = NamelistDefaults.alin
    clin: float = NamelistDefaults.clin

    @classmethod
    def from_f90nml(cls, namelist: f90nml.Namelist) -> "PhysicsConfig":
        namelist_dict = namelist_to_flatish_dict(namelist.items())
        namelist_dict = {
            key: value
            for key, value in namelist_dict.items()
            if key in cls.__dataclass_fields__  # type: ignore
        }
        return cls(**namelist_dict)
