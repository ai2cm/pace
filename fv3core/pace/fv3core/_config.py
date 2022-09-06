import dataclasses
from typing import Optional, Tuple

import f90nml

from pace.util import Namelist, NamelistDefaults


DEFAULT_INT = 0
DEFAULT_STR = ""
DEFAULT_FLOAT = 0.0
DEFAULT_BOOL = False


@dataclasses.dataclass(frozen=True)
class SatAdjustConfig:
    hydrostatic: bool
    rad_snow: bool
    rad_rain: bool
    rad_graupel: bool
    tintqs: bool
    sat_adj0: float
    ql_gen: float
    qs_mlt: float
    ql0_max: float
    t_sub: float
    qi_gen: float
    qi_lim: float
    qi0_max: float
    dw_ocean: float
    dw_land: float
    icloud_f: int
    cld_min: float
    tau_i2s: float
    tau_v2l: float
    tau_r2g: float
    tau_l2r: float
    tau_l2v: float
    tau_imlt: float
    tau_smlt: float


@dataclasses.dataclass(frozen=True)
class RemappingConfig:
    fill: bool
    kord_tm: int
    kord_tr: int
    kord_wz: int
    kord_mt: int
    do_sat_adj: bool
    sat_adjust: SatAdjustConfig

    @property
    def hydrostatic(self) -> bool:
        return self.sat_adjust.hydrostatic


@dataclasses.dataclass(frozen=True)
class RiemannConfig:
    p_fac: float
    a_imp: float
    use_logp: bool
    beta: float


@dataclasses.dataclass(frozen=True)
class DGridShallowWaterLagrangianDynamicsConfig:

    dddmp: float
    d2_bg: float
    d2_bg_k1: float
    d2_bg_k2: float
    d4_bg: float
    ke_bg: float
    nord: int
    n_sponge: int
    grid_type: int
    d_ext: float
    hord_dp: int
    hord_tm: int
    hord_mt: int
    hord_vt: int
    do_f3d: bool
    do_skeb: bool
    d_con: float
    vtdm4: float
    inline_q: bool
    convert_ke: bool
    do_vort_damp: bool
    hydrostatic: bool


@dataclasses.dataclass(frozen=True)
class AcousticDynamicsConfig:

    tau: float
    k_split: int
    n_split: int
    m_split: int
    delt_max: float
    rf_cutoff: float
    rf_fast: bool
    breed_vortex_inline: bool
    use_old_omega: bool
    riemann: RiemannConfig
    d_grid_shallow_water: DGridShallowWaterLagrangianDynamicsConfig

    @property
    def nord(self) -> int:
        return self.d_grid_shallow_water.nord

    @property
    def grid_type(self) -> int:
        return self.d_grid_shallow_water.grid_type

    @property
    def hydrostatic(self) -> bool:
        return self.d_grid_shallow_water.hydrostatic

    @property
    def hord_tm(self) -> int:
        return self.d_grid_shallow_water.hord_tm

    @property
    def p_fac(self) -> float:
        return self.riemann.p_fac

    @property
    def d_ext(self) -> float:
        return self.d_grid_shallow_water.d_ext

    @property
    def d_con(self) -> float:
        return self.d_grid_shallow_water.d_con

    @property
    def beta(self) -> float:
        return self.riemann.beta

    @property
    def use_logp(self) -> bool:
        return self.riemann.use_logp


@dataclasses.dataclass
class DynamicalCoreConfig:
    dt_atmos: int = DEFAULT_INT
    a_imp: float = DEFAULT_FLOAT
    beta: float = DEFAULT_FLOAT
    consv_te: float = DEFAULT_FLOAT
    d2_bg: float = DEFAULT_FLOAT
    d2_bg_k1: float = DEFAULT_FLOAT
    d2_bg_k2: float = DEFAULT_FLOAT
    d4_bg: float = DEFAULT_FLOAT
    d_con: float = DEFAULT_FLOAT
    d_ext: float = DEFAULT_FLOAT
    dddmp: float = DEFAULT_FLOAT
    delt_max: float = DEFAULT_FLOAT
    do_sat_adj: bool = DEFAULT_BOOL
    do_vort_damp: bool = DEFAULT_BOOL
    fill: bool = DEFAULT_BOOL
    hord_dp: int = DEFAULT_INT
    hord_mt: int = DEFAULT_INT
    hord_tm: int = DEFAULT_INT
    hord_tr: int = DEFAULT_INT
    hord_vt: int = DEFAULT_INT
    hydrostatic: bool = DEFAULT_BOOL
    k_split: int = DEFAULT_INT
    ke_bg: float = DEFAULT_FLOAT
    kord_mt: int = DEFAULT_INT
    kord_tm: int = DEFAULT_INT
    kord_tr: int = DEFAULT_INT
    kord_wz: int = DEFAULT_INT
    n_split: int = DEFAULT_INT
    nord: int = DEFAULT_INT
    npx: int = DEFAULT_INT
    npy: int = DEFAULT_INT
    npz: int = DEFAULT_INT
    ntiles: int = DEFAULT_INT
    nwat: int = DEFAULT_INT
    p_fac: float = DEFAULT_FLOAT
    rf_cutoff: float = DEFAULT_FLOAT
    tau: float = DEFAULT_FLOAT
    vtdm4: float = DEFAULT_FLOAT
    z_tracer: bool = DEFAULT_BOOL
    do_qa: bool = DEFAULT_BOOL
    layout: Tuple[int, int] = NamelistDefaults.layout
    grid_type: int = NamelistDefaults.grid_type
    do_f3d: bool = NamelistDefaults.do_f3d
    inline_q: bool = NamelistDefaults.inline_q
    do_skeb: bool = NamelistDefaults.do_skeb  # save dissipation estimate
    use_logp: bool = NamelistDefaults.use_logp
    moist_phys: bool = NamelistDefaults.moist_phys
    check_negative: bool = NamelistDefaults.check_negative
    # gfdl_cloud_microphys.F90
    tau_r2g: float = NamelistDefaults.tau_r2g  # rain freezing during fast_sat
    tau_smlt: float = NamelistDefaults.tau_smlt  # snow melting
    tau_g2r: float = NamelistDefaults.tau_g2r  # graupel melting to rain
    tau_imlt: float = NamelistDefaults.tau_imlt  # cloud ice melting
    tau_i2s: float = NamelistDefaults.tau_i2s  # cloud ice to snow auto - conversion
    tau_l2r: float = NamelistDefaults.tau_l2r  # cloud water to rain auto - conversion
    tau_g2v: float = NamelistDefaults.tau_g2v  # graupel sublimation
    tau_v2g: float = (
        NamelistDefaults.tau_v2g
    )  # graupel deposition -- make it a slow process
    sat_adj0: float = (
        NamelistDefaults.sat_adj0
    )  # adjustment factor (0: no 1: full) during fast_sat_adj
    ql_gen: float = (
        1.0e-3  # max new cloud water during remapping step if fast_sat_adj = .t.
    )
    ql_mlt: float = (
        NamelistDefaults.ql_mlt
    )  # max value of cloud water allowed from melted cloud ice
    qs_mlt: float = NamelistDefaults.qs_mlt  # max cloud water due to snow melt
    ql0_max: float = (
        NamelistDefaults.ql0_max
    )  # max cloud water value (auto converted to rain)
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
    rad_graupel: bool = (
        NamelistDefaults.rad_graupel
    )  # consider graupel in cloud fraction calculation
    tintqs: bool = (
        NamelistDefaults.tintqs
    )  # use temperature in the saturation mixing in PDF
    dw_ocean: float = NamelistDefaults.dw_ocean  # base value for ocean
    dw_land: float = (
        NamelistDefaults.dw_land
    )  # base value for subgrid deviation / variability over land
    # cloud scheme 0 - ?
    # 1: old fvgfs gfdl) mp implementation
    # 2: binary cloud scheme (0 / 1)
    icloud_f: int = NamelistDefaults.icloud_f
    cld_min: float = NamelistDefaults.cld_min  # !< minimum cloud fraction
    tau_l2v: float = (
        NamelistDefaults.tau_l2v
    )  # cloud water to water vapor (evaporation)
    tau_v2l: float = (
        NamelistDefaults.tau_v2l
    )  # water vapor to cloud water (condensation)
    c2l_ord: int = NamelistDefaults.c2l_ord
    regional: bool = NamelistDefaults.regional
    m_split: int = NamelistDefaults.m_split
    convert_ke: bool = NamelistDefaults.convert_ke
    breed_vortex_inline: bool = NamelistDefaults.breed_vortex_inline
    use_old_omega: bool = NamelistDefaults.use_old_omega
    rf_fast: bool = NamelistDefaults.rf_fast
    p_ref: float = (
        NamelistDefaults.p_ref
    )  # Surface pressure used to construct a horizontally-uniform reference
    adiabatic: bool = NamelistDefaults.adiabatic
    nf_omega: int = NamelistDefaults.nf_omega
    fv_sg_adj: int = NamelistDefaults.fv_sg_adj
    n_sponge: int = NamelistDefaults.n_sponge
    namelist_override: Optional[str] = None

    def __post_init__(self):
        if self.namelist_override is not None:
            try:
                f90_nml = f90nml.read(self.namelist_override)
            except FileNotFoundError:
                print(f"{self.namelist_override} does not exist")
                raise
            dycore_config = self.from_f90nml(f90_nml)
            for var in dycore_config.__dict__.keys():
                setattr(self, var, dycore_config.__dict__[var])

    @classmethod
    def from_f90nml(self, f90_namelist: f90nml.Namelist) -> "DynamicalCoreConfig":
        namelist = Namelist.from_f90nml(f90_namelist)
        return self.from_namelist(namelist)

    @classmethod
    def from_namelist(cls, namelist: Namelist) -> "DynamicalCoreConfig":
        return cls(
            dt_atmos=namelist.dt_atmos,
            a_imp=namelist.a_imp,
            beta=namelist.beta,
            consv_te=namelist.consv_te,
            d2_bg=namelist.d2_bg,
            d2_bg_k1=namelist.d2_bg_k1,
            d2_bg_k2=namelist.d2_bg_k2,
            d4_bg=namelist.d4_bg,
            d_con=namelist.d_con,
            d_ext=namelist.d_ext,
            dddmp=namelist.dddmp,
            delt_max=namelist.delt_max,
            do_sat_adj=namelist.do_sat_adj,
            do_vort_damp=namelist.do_vort_damp,
            fill=namelist.fill,
            hord_dp=namelist.hord_dp,
            hord_mt=namelist.hord_mt,
            hord_tm=namelist.hord_tm,
            hord_tr=namelist.hord_tr,
            hord_vt=namelist.hord_vt,
            hydrostatic=namelist.hydrostatic,
            k_split=namelist.k_split,
            ke_bg=namelist.ke_bg,
            kord_mt=namelist.kord_mt,
            kord_tm=namelist.kord_tm,
            kord_tr=namelist.kord_tr,
            kord_wz=namelist.kord_wz,
            n_split=namelist.n_split,
            nord=namelist.nord,
            npx=namelist.npx,
            npy=namelist.npy,
            npz=namelist.npz,
            ntiles=namelist.ntiles,
            nwat=namelist.nwat,
            p_fac=namelist.p_fac,
            rf_cutoff=namelist.rf_cutoff,
            tau=namelist.tau,
            vtdm4=namelist.vtdm4,
            z_tracer=namelist.z_tracer,
            do_qa=namelist.do_qa,
            layout=namelist.layout,
            grid_type=namelist.grid_type,
            do_f3d=namelist.do_f3d,
            inline_q=namelist.inline_q,
            do_skeb=namelist.do_skeb,
            check_negative=namelist.check_negative,
            tau_r2g=namelist.tau_r2g,
            tau_smlt=namelist.tau_smlt,
            tau_g2r=namelist.tau_g2r,
            tau_imlt=namelist.tau_imlt,
            tau_i2s=namelist.tau_i2s,
            tau_l2r=namelist.tau_l2r,
            tau_g2v=namelist.tau_g2v,
            tau_v2g=namelist.tau_v2g,
            sat_adj0=namelist.sat_adj0,
            ql_gen=namelist.ql_gen,
            ql_mlt=namelist.ql_mlt,
            qs_mlt=namelist.qs_mlt,
            ql0_max=namelist.ql0_max,
            t_sub=namelist.t_sub,
            qi_gen=namelist.qi_gen,
            qi_lim=namelist.qi_lim,
            qi0_max=namelist.qi0_max,
            rad_snow=namelist.rad_snow,
            rad_rain=namelist.rad_rain,
            rad_graupel=namelist.rad_graupel,
            tintqs=namelist.tintqs,
            dw_ocean=namelist.dw_ocean,
            dw_land=namelist.dw_land,
            icloud_f=namelist.icloud_f,
            cld_min=namelist.cld_min,
            tau_l2v=namelist.tau_l2v,
            tau_v2l=namelist.tau_v2l,
            c2l_ord=namelist.c2l_ord,
            regional=namelist.regional,
            m_split=namelist.m_split,
            convert_ke=namelist.convert_ke,
            breed_vortex_inline=namelist.breed_vortex_inline,
            use_old_omega=namelist.use_old_omega,
            rf_fast=namelist.rf_fast,
            p_ref=namelist.p_ref,
            adiabatic=namelist.adiabatic,
            nf_omega=namelist.nf_omega,
            fv_sg_adj=namelist.fv_sg_adj,
            n_sponge=namelist.n_sponge,
        )

    @property
    def do_dry_convective_adjustment(self) -> bool:
        return self.fv_sg_adj > 0

    @property
    def riemann(self) -> RiemannConfig:
        return RiemannConfig(
            p_fac=self.p_fac,
            a_imp=self.a_imp,
            use_logp=self.use_logp,
            beta=self.beta,
        )

    @property
    def d_grid_shallow_water(self) -> DGridShallowWaterLagrangianDynamicsConfig:
        return DGridShallowWaterLagrangianDynamicsConfig(
            dddmp=self.dddmp,
            d2_bg=self.d2_bg,
            d2_bg_k1=self.d2_bg_k1,
            d2_bg_k2=self.d2_bg_k2,
            d4_bg=self.d4_bg,
            ke_bg=self.ke_bg,
            nord=self.nord,
            n_sponge=self.n_sponge,
            grid_type=self.grid_type,
            d_ext=self.d_ext,
            inline_q=self.inline_q,
            hord_dp=self.hord_dp,
            hord_tm=self.hord_tm,
            hord_mt=self.hord_mt,
            hord_vt=self.hord_vt,
            do_f3d=self.do_f3d,
            do_skeb=self.do_skeb,
            d_con=self.d_con,
            vtdm4=self.vtdm4,
            do_vort_damp=self.do_vort_damp,
            hydrostatic=self.hydrostatic,
            convert_ke=self.convert_ke,
        )

    @property
    def acoustic_dynamics(self) -> AcousticDynamicsConfig:
        return AcousticDynamicsConfig(
            tau=self.tau,
            k_split=self.k_split,
            n_split=self.n_split,
            m_split=self.m_split,
            delt_max=self.delt_max,
            rf_fast=self.rf_fast,
            rf_cutoff=self.rf_cutoff,
            breed_vortex_inline=self.breed_vortex_inline,
            use_old_omega=self.use_old_omega,
            riemann=self.riemann,
            d_grid_shallow_water=self.d_grid_shallow_water,
        )

    @property
    def sat_adjust(self) -> SatAdjustConfig:
        return SatAdjustConfig(
            hydrostatic=self.hydrostatic,
            rad_snow=self.rad_snow,
            rad_rain=self.rad_rain,
            rad_graupel=self.rad_graupel,
            tintqs=self.tintqs,
            sat_adj0=self.sat_adj0,
            ql_gen=self.ql_gen,
            qs_mlt=self.qs_mlt,
            ql0_max=self.ql0_max,
            t_sub=self.t_sub,
            qi_gen=self.qi_gen,
            qi_lim=self.qi_lim,
            qi0_max=self.qi0_max,
            dw_ocean=self.dw_ocean,
            dw_land=self.dw_land,
            icloud_f=self.icloud_f,
            cld_min=self.cld_min,
            tau_i2s=self.tau_i2s,
            tau_v2l=self.tau_v2l,
            tau_r2g=self.tau_r2g,
            tau_l2r=self.tau_l2r,
            tau_l2v=self.tau_l2v,
            tau_imlt=self.tau_imlt,
            tau_smlt=self.tau_smlt,
        )

    @property
    def remapping(self) -> RemappingConfig:
        return RemappingConfig(
            fill=self.fill,
            kord_tm=self.kord_tm,
            kord_tr=self.kord_tr,
            kord_wz=self.kord_wz,
            kord_mt=self.kord_mt,
            do_sat_adj=self.do_sat_adj,
            sat_adjust=self.sat_adjust,
        )
