import dataclasses
import os
from typing import Tuple

import f90nml

import fv3core.utils.gt4py_utils as utils
from fv3core.utils.grid import Grid


grid = None

# Global set of namelist defaults
namelist_defaults = {
    "grid_type": 0,
    "do_f3d": False,
    "inline_q": False,
    "do_skeb": False,  # save dissipation estimate
    "use_logp": False,
    "moist_phys": True,
    "check_negative": False,
    # gfdl_cloud_mucrophys.F90
    "tau_r2g": 900.0,  # rain freezing during fast_sat
    "tau_smlt": 900.0,  # snow melting
    "tau_g2r": 600.0,  # graupel melting to rain
    "tau_imlt": 600.0,  # cloud ice melting
    "tau_i2s": 1000.0,  # cloud ice to snow auto - conversion
    "tau_l2r": 900.0,  # cloud water to rain auto - conversion
    "tau_g2v": 900.0,  # graupel sublimation
    "tau_v2g": 21600.0,  # graupel deposition -- make it a slow process
    "sat_adj0": 0.90,  # adjustment factor (0: no, 1: full) during fast_sat_adj
    "ql_gen": 1.0e-3,  # max new cloud water during remapping step if fast_sat_adj = .t.
    "ql_mlt": 2.0e-3,  # max value of cloud water allowed from melted cloud ice
    "qs_mlt": 1.0e-6,  # max cloud water due to snow melt
    "ql0_max": 2.0e-3,  # max cloud water value (auto converted to rain)
    "t_sub": 184.0,  # min temp for sublimation of cloud ice
    "qi_gen": 1.82e-6,  # max cloud ice generation during remapping step
    "qi_lim": 1.0,  # cloud ice limiter to prevent large ice build up
    "qi0_max": 1.0e-4,  # max cloud ice value (by other sources)
    "rad_snow": True,  # consider snow in cloud fraciton calculation
    "rad_rain": True,  # consider rain in cloud fraction calculation
    "rad_graupel": True,  # consider graupel in cloud fraction calculation
    "tintqs": False,  # use temperature in the saturation mixing in PDF
    "dw_ocean": 0.10,  # base value for ocean
    "dw_land": 0.20,  # base value for subgrid deviation / variability over land
    # cloud scheme 0 - ?
    # 1: old fvgfs gfdl) mp implementation
    # 2: binary cloud scheme (0 / 1)
    "icloud_f": 0,
    "cld_min": 0.05,  # !< minimum cloud fraction
    "tau_l2v": 300.0,  # cloud water to water vapor (evaporation)
    "tau_v2l": 150.0,  # water vapor to cloud water (condensation)
    "c2l_ord": 4,
    "regional": False,
    "m_split": 0,
    "convert_ke": False,
    "breed_vortex_inline": False,
    "use_old_omega": True,
    "use_logp": False,
    "RF_fast": False,
    "p_ref": 1e5,  # Surface pressure used to construct a horizontally-uniform reference
    "adiabatic": False,
    "nf_omega": 1,
    "fv_sg_adj": -1,
    "n_sponge": 1,
}

# we need defaults for everything because of global state, these non-sensical defaults
# can be removed when global state is no longer used
DEFAULT_INT = 0
DEFAULT_STR = ""
DEFAULT_FLOAT = 0.0
DEFAULT_BOOL = False


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
    # dycore_only: bool
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
    # c_cracw: Any
    # c_paut: Any
    # c_pgacs: Any
    # c_psaci: Any
    # ccn_l: Any
    # ccn_o: Any
    # const_vg: bool
    # const_vi: bool
    # const_vr: bool
    # const_vs: bool
    # de_ice: Any
    do_qa: bool = DEFAULT_BOOL
    # do_sedi_heat: Any
    # do_sedi_w: Any
    # fast_sat_adj: bool
    # fix_negative: bool
    # irain_f: Any
    # mono_prof: Any
    # mp_time: Any
    # prog_ccn: Any
    # qi0_crt: Any
    # qs0_crt: Any
    # rh_inc: Any
    # rh_inr: Any
    # rh_ins: Any
    # rthresh: Any
    # sedi_transport: Any
    # use_ccn: Any
    # use_ppm: Any
    # vg_max: Any
    # vi_max: Any
    # vr_max: Any
    # vs_max: Any
    # z_slope_ice: Any
    # z_slope_liq: Any
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
    grid_type: int = 0
    do_f3d: bool = False
    inline_q: bool = False
    do_skeb: bool = False  # save dissipation estimate
    use_logp: bool = False
    moist_phys: bool = True
    check_negative: bool = False
    # gfdl_cloud_microphys.F90
    tau_r2g: float = 900.0  # rain freezing during fast_sat
    tau_smlt: float = 900.0  # snow melting
    tau_g2r: float = 600.0  # graupel melting to rain
    tau_imlt: float = 600.0  # cloud ice melting
    tau_i2s: float = 1000.0  # cloud ice to snow auto - conversion
    tau_l2r: float = 900.0  # cloud water to rain auto - conversion
    tau_g2v: float = 900.0  # graupel sublimation
    tau_v2g: float = 21600.0  # graupel deposition -- make it a slow process
    sat_adj0: float = 0.90  # adjustment factor (0: no 1: full) during fast_sat_adj
    ql_gen: float = (
        1.0e-3  # max new cloud water during remapping step if fast_sat_adj = .t.
    )
    ql_mlt: float = 2.0e-3  # max value of cloud water allowed from melted cloud ice
    qs_mlt: float = 1.0e-6  # max cloud water due to snow melt
    ql0_max: float = 2.0e-3  # max cloud water value (auto converted to rain)
    t_sub: float = 184.0  # min temp for sublimation of cloud ice
    qi_gen: float = 1.82e-6  # max cloud ice generation during remapping step
    qi_lim: float = 1.0  # cloud ice limiter to prevent large ice build up
    qi0_max: float = 1.0e-4  # max cloud ice value (by other sources)
    rad_snow: bool = True  # consider snow in cloud fraciton calculation
    rad_rain: bool = True  # consider rain in cloud fraction calculation
    rad_graupel: bool = True  # consider graupel in cloud fraction calculation
    tintqs: bool = False  # use temperature in the saturation mixing in PDF
    dw_ocean: float = 0.10  # base value for ocean
    dw_land: float = 0.20  # base value for subgrid deviation / variability over land
    # cloud scheme 0 - ?
    # 1: old fvgfs gfdl) mp implementation
    # 2: binary cloud scheme (0 / 1)
    icloud_f: int = 0
    cld_min: float = 0.05  # !< minimum cloud fraction
    tau_l2v: float = 300.0  # cloud water to water vapor (evaporation)
    tau_v2l: float = 150.0  # water vapor to cloud water (condensation)
    c2l_ord: int = 4
    regional: bool = False
    m_split: int = 0
    convert_ke: bool = False
    breed_vortex_inline: bool = False
    use_old_omega: bool = True
    rf_fast: bool = False
    p_ref: float = (
        1e5  # Surface pressure used to construct a horizontally-uniform reference
    )
    adiabatic: bool = False
    nf_omega: int = 1
    fv_sg_adj: int = -1
    n_sponge: int = 1

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


namelist = Namelist()


def namelist_to_flatish_dict(nml_input):
    nml = dict(nml_input)
    for name, value in nml.items():
        if isinstance(value, f90nml.Namelist):
            nml[name] = namelist_to_flatish_dict(value)
    flatter_namelist = {}
    for key, value in nml.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if subkey in flatter_namelist:
                    raise Exception(
                        "Cannot flatten this namelist, duplicate keys: " + subkey
                    )
                flatter_namelist[subkey] = subvalue
        else:
            flatter_namelist[key] = value
    return flatter_namelist


# TODO: Before this can be used, we need to write a module to make the grid data
# from files on disk and call it
def make_grid_from_namelist(namelist, rank):
    shape_params = {}
    for narg in ["npx", "npy", "npz"]:
        shape_params[narg] = getattr(namelist, narg)
    indices = {
        "isd": 0,
        "ied": namelist.npx + 2 * utils.halo - 2,
        "is_": utils.halo,
        "ie": namelist.npx + utils.halo - 2,
        "jsd": 0,
        "jed": namelist.npy + 2 * utils.halo - 2,
        "js": utils.halo,
        "je": namelist.npy + utils.halo - 2,
    }
    return Grid(indices, shape_params, rank, namelist.layout)


def set_grid(in_grid):
    """Updates the global grid given another.

    Args:
        in_grid (Grid): Input grid to set.
    """
    global grid
    grid = in_grid


def set_namelist(filename):
    """Updates the global namelist from a file and re-generates the global grid.

    Args:
        filename (str): Input file.
    """
    global grid
    namelist_dict = namelist_defaults.copy()
    namelist_dict.update(namelist_to_flatish_dict(f90nml.read(filename).items()))
    for name, value in namelist_dict.items():
        setattr(namelist, name, value)

    grid = make_grid_from_namelist(namelist, 0)


if "NAMELIST_FILENAME" in os.environ:
    set_namelist(os.environ["NAMELIST_FILENAME"])
