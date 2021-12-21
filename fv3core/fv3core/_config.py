import dataclasses
from typing import Tuple

import f90nml

import pace.dsl.gt4py_utils as utils
from pace.stencils.testing.grid import Grid
from pace.util.namelist import NamelistDefaults, namelist_to_flatish_dict


grid = None

# we need defaults for everything because of global state, these non-sensical defaults
# can be removed when global state is no longer used
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
    dt_atmos: int
    a_imp: float
    beta: float
    consv_te: bool
    d2_bg: float
    d2_bg_k1: float
    d2_bg_k2: float
    d4_bg: float
    d_con: float
    d_ext: float
    dddmp: float
    delt_max: float
    do_sat_adj: bool
    do_vort_damp: bool
    fill: bool
    hord_dp: int
    hord_mt: int
    hord_tm: int
    hord_tr: int
    hord_vt: int
    hydrostatic: bool
    k_split: int
    ke_bg: float
    kord_mt: int
    kord_tm: int
    kord_tr: int
    kord_wz: int
    n_split: int
    nord: int
    npx: int
    npy: int
    npz: int
    ntiles: int
    nwat: int
    p_fac: float
    rf_cutoff: float
    tau: float
    vtdm4: float
    z_tracer: bool
    do_qa: bool
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

    @classmethod
    def from_f90nml(cls, namelist: f90nml.Namelist) -> "DynamicalCoreConfig":
        namelist_dict = namelist_to_flatish_dict(namelist.items())
        namelist_dict = {
            key: value
            for key, value in namelist_dict.items()
            if key in cls.__dataclass_fields__  # type: ignore
        }
        return cls(**namelist_dict)

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


@dataclasses.dataclass
class Namelist:
    # data_set: Any
    # date_out_of_range: str
    # do_sst_pert: bool
    # interp_oi_sst: bool
    # no_anom_sst: bool
    # sst_pert: float
    # sst_pert_type: str
    # use_daily: bool
    # use_ncep_ice: bool
    # use_ncep_sst: bool
    # blocksize: int
    # chksum_debug: bool
    """
    WARNING: dycore_only may not be used in this model
     the same way it is in the Fortran version, watch for
     consequences of these inconsistencies, or more closely
     parallel the Fortran structure
    """
    dycore_only: bool = DEFAULT_BOOL
    # fdiag: float
    # knob_ugwp_azdir: Tuple[int, int, int, int]
    # knob_ugwp_doaxyz: int
    # knob_ugwp_doheat: int
    # knob_ugwp_dokdis: int
    # knob_ugwp_effac: Tuple[int, int, int, int]
    # knob_ugwp_ndx4lh: int
    # knob_ugwp_solver: int
    # knob_ugwp_source: Tuple[int, int, int, int]
    # knob_ugwp_stoch: Tuple[int, int, int, int]
    # knob_ugwp_version: int
    # knob_ugwp_wvspec: Tuple[int, int, int, int]
    # launch_level: int
    # reiflag: int
    # reimax: float
    # reimin: float
    # rewmax: float
    # rewmin: float
    # atmos_nthreads: int
    # calendar: Any
    # current_date: Any
    # days: Any
    dt_atmos: int = DEFAULT_INT
    # dt_ocean: Any
    # hours: Any
    # memuse_verbose: Any
    # minutes: Any
    # months: Any
    # ncores_per_node: Any
    # seconds: Any
    # use_hyper_thread: Any
    # max_axes: Any
    # max_files: Any
    # max_num_axis_sets: Any
    # prepend_date: Any
    # checker_tr: Any
    # filtered_terrain: Any
    # gfs_dwinds: Any
    # levp: Any
    # nt_checker: Any
    # checksum_required: Any
    # max_files_r: Any
    # max_files_w: Any
    # clock_grain: Any
    # domains_stack_size: Any
    # print_memory_usage: Any
    a_imp: float = DEFAULT_FLOAT
    # adjust_dry_mass: Any
    beta: float = DEFAULT_FLOAT
    # consv_am: Any
    consv_te: bool = DEFAULT_BOOL
    d2_bg: float = DEFAULT_FLOAT
    d2_bg_k1: float = DEFAULT_FLOAT
    d2_bg_k2: float = DEFAULT_FLOAT
    d4_bg: float = DEFAULT_FLOAT
    d_con: float = DEFAULT_FLOAT
    d_ext: float = DEFAULT_FLOAT
    dddmp: float = DEFAULT_FLOAT
    delt_max: float = DEFAULT_FLOAT
    # dnats: int
    do_sat_adj: bool = DEFAULT_BOOL
    do_vort_damp: bool = DEFAULT_BOOL
    # dwind_2d: Any
    # external_ic: Any
    fill: bool = DEFAULT_BOOL
    # fill_dp: bool
    # fv_debug: Any
    # gfs_phil: Any
    hord_dp: int = DEFAULT_INT
    hord_mt: int = DEFAULT_INT
    hord_tm: int = DEFAULT_INT
    hord_tr: int = DEFAULT_INT
    hord_vt: int = DEFAULT_INT
    hydrostatic: bool = DEFAULT_BOOL
    # io_layout: Any
    k_split: int = DEFAULT_INT
    ke_bg: float = DEFAULT_FLOAT
    kord_mt: int = DEFAULT_INT
    kord_tm: int = DEFAULT_INT
    kord_tr: int = DEFAULT_INT
    kord_wz: int = DEFAULT_INT
    layout: Tuple[int, int] = (1, 1)
    # make_nh: bool
    # mountain: bool
    n_split: int = DEFAULT_INT
    # na_init: Any
    # ncep_ic: Any
    # nggps_ic: Any
    nord: int = DEFAULT_INT
    npx: int = DEFAULT_INT
    npy: int = DEFAULT_INT
    npz: int = DEFAULT_INT
    ntiles: int = DEFAULT_INT
    # nudge: Any
    # nudge_qv: Any
    nwat: int = DEFAULT_INT
    p_fac: float = DEFAULT_FLOAT
    # phys_hydrostatic: Any
    # print_freq: Any
    # range_warn: Any
    # reset_eta: Any
    rf_cutoff: float = DEFAULT_FLOAT
    tau: float = DEFAULT_FLOAT
    # tau_h2o: Any
    # use_hydro_pressure: Any
    vtdm4: float = DEFAULT_FLOAT
    # warm_start: bool
    z_tracer: bool = DEFAULT_BOOL
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
    qc_crt: float = NamelistDefaults.qc_crt
    vs_fac: float = NamelistDefaults.vs_fac
    vg_fac: float = NamelistDefaults.vg_fac
    vi_fac: float = NamelistDefaults.vi_fac
    vr_fac: float = NamelistDefaults.vr_fac
    de_ice: bool = NamelistDefaults.de_ice
    do_qa: bool = NamelistDefaults.do_qa
    do_sedi_heat: bool = NamelistDefaults.do_sedi_heat
    do_sedi_w: bool = NamelistDefaults.do_sedi_w
    fast_sat_adj: bool = NamelistDefaults.fast_sat_adj
    fix_negative: bool = NamelistDefaults.fix_negative
    irain_f: int = NamelistDefaults.irain_f
    mono_prof: bool = NamelistDefaults.mono_prof
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
    # c0s_shal: Any
    # c1_shal: Any
    # cal_pre: Any
    # cdmbgwd: Any
    # cnvcld: Any
    # cnvgwd: Any
    # debug: Any
    # do_deep: Any
    # dspheat: Any
    # fhcyc: Any
    # fhlwr: Any
    # fhswr: Any
    # fhzero: Any
    # hybedmf: Any
    # iaer: Any
    # ialb: Any
    # ico2: Any
    # iems: Any
    # imfdeepcnv: Any
    # imfshalcnv: Any
    # imp_physics: Any
    # isol: Any
    # isot: Any
    # isubc_lw: Any
    # isubc_sw: Any
    # ivegsrc: Any
    # ldiag3d: Any
    # lwhtr: Any
    # ncld: int
    # nst_anl: Any
    # pdfcld: Any
    # pre_rad: Any
    # prslrd0: Any
    # random_clds: Any
    # redrag: Any
    # satmedmf: Any
    # shal_cnv: Any
    # swhtr: Any
    # trans_trac: Any
    # use_ufo: Any
    # xkzm_h: Any
    # xkzm_m: Any
    # xkzminv: Any
    # interp_method: Any
    # lat_s: Any
    # lon_s: Any
    # ntrunc: Any
    # fabsl: Any
    # faisl: Any
    # faiss: Any
    # fnabsc: Any
    # fnacna: Any
    # fnaisc: Any
    # fnalbc: Any
    # fnalbc2: Any
    # fnglac: Any
    # fnmskh: Any
    # fnmxic: Any
    # fnslpc: Any
    # fnsmcc: Any
    # fnsnoa: Any
    # fnsnoc: Any
    # fnsotc: Any
    # fntg3c: Any
    # fntsfa: Any
    # fntsfc: Any
    # fnvegc: Any
    # fnvetc: Any
    # fnvmnc: Any
    # fnvmxc: Any
    # fnzorc: Any
    # fsicl: Any
    # fsics: Any
    # fslpl: Any
    # fsmcl: Any
    # fsnol: Any
    # fsnos: Any
    # fsotl: Any
    # ftsfl: Any
    # ftsfs: Any
    # fvetl: Any
    # fvmnl: Any
    # fvmxl: Any
    # ldebug: Any
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

    @classmethod
    def from_f90nml(cls, namelist: f90nml.Namelist):
        namelist_dict = namelist_to_flatish_dict(namelist.items())
        namelist_dict = {
            key: value
            for key, value in namelist_dict.items()
            if key in cls.__dataclass_fields__  # type: ignore
        }
        return cls(**namelist_dict)

    @property
    def dynamical_core(self) -> DynamicalCoreConfig:
        return DynamicalCoreConfig(
            dt_atmos=self.dt_atmos,
            a_imp=self.a_imp,
            beta=self.beta,
            consv_te=self.consv_te,
            d2_bg=self.d2_bg,
            d2_bg_k1=self.d2_bg_k1,
            d2_bg_k2=self.d2_bg_k2,
            d4_bg=self.d4_bg,
            d_con=self.d_con,
            d_ext=self.d_ext,
            dddmp=self.dddmp,
            delt_max=self.delt_max,
            do_sat_adj=self.do_sat_adj,
            do_vort_damp=self.do_vort_damp,
            fill=self.fill,
            hord_dp=self.hord_dp,
            hord_mt=self.hord_mt,
            hord_tm=self.hord_tm,
            hord_tr=self.hord_tr,
            hord_vt=self.hord_vt,
            hydrostatic=self.hydrostatic,
            k_split=self.k_split,
            ke_bg=self.ke_bg,
            kord_mt=self.kord_mt,
            kord_tm=self.kord_tm,
            kord_tr=self.kord_tr,
            kord_wz=self.kord_wz,
            n_split=self.n_split,
            nord=self.nord,
            npx=self.npx,
            npy=self.npy,
            npz=self.npz,
            ntiles=self.ntiles,
            nwat=self.nwat,
            p_fac=self.p_fac,
            rf_cutoff=self.rf_cutoff,
            tau=self.tau,
            vtdm4=self.vtdm4,
            z_tracer=self.z_tracer,
            do_qa=self.do_qa,
            layout=self.layout,
            grid_type=self.grid_type,
            do_f3d=self.do_f3d,
            inline_q=self.inline_q,
            do_skeb=self.do_skeb,
            check_negative=self.check_negative,
            tau_r2g=self.tau_r2g,
            tau_smlt=self.tau_smlt,
            tau_g2r=self.tau_g2r,
            tau_imlt=self.tau_imlt,
            tau_i2s=self.tau_i2s,
            tau_l2r=self.tau_l2r,
            tau_g2v=self.tau_g2v,
            tau_v2g=self.tau_v2g,
            sat_adj0=self.sat_adj0,
            ql_gen=self.ql_gen,
            ql_mlt=self.ql_mlt,
            qs_mlt=self.qs_mlt,
            ql0_max=self.ql0_max,
            t_sub=self.t_sub,
            qi_gen=self.qi_gen,
            qi_lim=self.qi_lim,
            qi0_max=self.qi0_max,
            rad_snow=self.rad_snow,
            rad_rain=self.rad_rain,
            rad_graupel=self.rad_graupel,
            tintqs=self.tintqs,
            dw_ocean=self.dw_ocean,
            dw_land=self.dw_land,
            icloud_f=self.icloud_f,
            cld_min=self.cld_min,
            tau_l2v=self.tau_l2v,
            tau_v2l=self.tau_v2l,
            c2l_ord=self.c2l_ord,
            regional=self.regional,
            m_split=self.m_split,
            convert_ke=self.convert_ke,
            breed_vortex_inline=self.breed_vortex_inline,
            use_old_omega=self.use_old_omega,
            rf_fast=self.rf_fast,
            p_ref=self.p_ref,
            adiabatic=self.adiabatic,
            nf_omega=self.nf_omega,
            fv_sg_adj=self.fv_sg_adj,
            n_sponge=self.n_sponge,
        )


namelist = Namelist()


def make_grid_from_namelist(namelist, rank, backend):
    shape_params = {}
    for narg in ["npx", "npy", "npz"]:
        shape_params[narg] = getattr(namelist, narg)
    # TODO this won't work with variable sized domains
    # but this entire method will be refactored away
    # and not used soon
    nx = int((namelist.npx - 1) / namelist.layout[0])
    ny = int((namelist.npy - 1) / namelist.layout[1])
    indices = {
        "isd": 0,
        "ied": nx + 2 * utils.halo - 1,
        "is_": utils.halo,
        "ie": nx + utils.halo - 1,
        "jsd": 0,
        "jed": ny + 2 * utils.halo - 1,
        "js": utils.halo,
        "je": ny + utils.halo - 1,
    }
    return Grid(
        indices, shape_params, rank, namelist.layout, backend, local_indices=True
    )


def make_grid_with_data_from_namelist(namelist, communicator, backend):
    grid = make_grid_from_namelist(namelist, communicator.rank, backend)
    grid.make_grid_data(
        npx=namelist.npx,
        npy=namelist.npy,
        npz=namelist.npz,
        communicator=communicator,
        backend=backend,
    )
    return grid


def set_grid(in_grid):
    """Updates the global grid given another.

    Args:
        in_grid (Grid): Input grid to set.
    """
    global grid
    grid = in_grid
