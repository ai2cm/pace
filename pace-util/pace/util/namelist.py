import f90nml


# Global set of namelist defaults, attached to class for namespacing and static typing
class NamelistDefaults:
    layout = (1, 1)
    grid_type = 0
    do_f3d = False
    inline_q = False
    do_skeb = False  # save dissipation estimate
    use_logp = False
    moist_phys = True
    check_negative = False
    # gfdl_cloud_mucrophys.F90
    tau_r2g = 900.0  # rain freezing during fast_sat
    tau_smlt = 900.0  # snow melting
    tau_g2r = 600.0  # graupel melting to rain
    tau_imlt = 600.0  # cloud ice melting
    tau_i2s = 1000.0  # cloud ice to snow auto - conversion
    tau_l2r = 900.0  # cloud water to rain auto - conversion
    tau_g2v = 900.0  # graupel sublimation
    tau_v2g = 21600.0  # graupel deposition -- make it a slow process
    sat_adj0 = 0.90  # adjustment factor (0: no, 1: full) during fast_sat_adj
    ql_gen = 1.0e-3  # max new cloud water during remapping step if fast_sat_adj = .t.
    ql_mlt = 2.0e-3  # max value of cloud water allowed from melted cloud ice
    qs_mlt = 1.0e-6  # max cloud water due to snow melt
    ql0_max = 2.0e-3  # max cloud water value (auto converted to rain)
    t_sub = 184.0  # min temp for sublimation of cloud ice
    qi_gen = 1.82e-6  # max cloud ice generation during remapping step
    qi_lim = 1.0  # cloud ice limiter to prevent large ice build up
    qi0_max = 1.0e-4  # max cloud ice value (by other sources)
    rad_snow = True  # consider snow in cloud fraciton calculation
    rad_rain = True  # consider rain in cloud fraction calculation
    rad_graupel = True  # consider graupel in cloud fraction calculation
    tintqs = False  # use temperature in the saturation mixing in PDF
    dw_ocean = 0.10  # base value for ocean
    dw_land = 0.20  # base value for subgrid deviation / variability over land
    # cloud scheme 0 - ?
    # 1: old fvgfs gfdl) mp implementation
    # 2: binary cloud scheme (0 / 1)
    icloud_f = 0
    cld_min = 0.05  # !< minimum cloud fraction
    tau_l2v = 300.0  # cloud water to water vapor (evaporation)
    tau_v2l = 150.0  # water vapor to cloud water (condensation)
    c2l_ord = 4
    regional = False
    m_split = 0
    convert_ke = False
    breed_vortex_inline = False
    use_old_omega = True
    use_logp = False
    rf_fast = False
    p_ref = 1e5  # Surface pressure used to construct a horizontally-uniform reference
    adiabatic = False
    nf_omega = 1
    fv_sg_adj = -1
    n_sponge = 1
    fast_sat_adj = False
    qc_crt = 5.0e-8  # Minimum condensate mixing ratio to allow partial cloudiness
    c_cracw = 0.8  # Rain accretion efficiency
    c_paut = (
        0.5  # Autoconversion cloud water to rain (use 0.5 to reduce autoconversion)
    )
    c_pgacs = 0.01  # Snow to graupel "accretion" eff. (was 0.1 in zetac)
    c_psaci = 0.05  # Accretion: cloud ice to snow (was 0.1 in zetac)
    ccn_l = 300.0  # CCN over land (cm^-3)
    ccn_o = 100.0  # CCN over ocean (cm^-3)
    const_vg = False  # Fall velocity tuning constant of graupel
    const_vi = False  # Fall velocity tuning constant of ice
    const_vr = False  # Fall velocity tuning constant of rain water
    const_vs = False  # Fall velocity tuning constant of snow
    vi_fac = 1.0  # if const_vi: 1/3
    vs_fac = 1.0  # if const_vs: 1.
    vg_fac = 1.0  # if const_vg: 2.
    vr_fac = 1.0  # if const_vr: 4.
    de_ice = False  # To prevent excessive build-up of cloud ice from external sources
    do_qa = True  # Do inline cloud fraction
    do_sedi_heat = False  # Transport of heat in sedimentation
    do_sedi_w = False  # Transport of vertical motion in sedimentation
    fix_negative = True  # Fix negative water species
    irain_f = 0  # Cloud water to rain auto conversion scheme
    mono_prof = False  # Perform terminal fall with mono ppm scheme
    mp_time = 225.0  # Maximum microphysics timestep (sec)
    prog_ccn = False  # Do prognostic ccn (yi ming's method)
    qi0_crt = 8e-05  # Cloud ice to snow autoconversion threshold
    qs0_crt = 0.003  # Snow to graupel density threshold (0.6e-3 in purdue lin scheme)
    rh_inc = 0.2  # RH increment for complete evaporation of cloud water and cloud ice
    rh_inr = 0.3  # RH increment for minimum evaporation of rain
    rthresh = 1e-05  # Critical cloud drop radius (micrometers)
    sedi_transport = True  # Transport of momentum in sedimentation
    use_ppm = False  # Use ppm fall scheme
    vg_max = 16.0  # Maximum fall speed for graupel
    vi_max = 1.0  # Maximum fall speed for ice
    vr_max = 16.0  # Maximum fall speed for rain
    vs_max = 2.0  # Maximum fall speed for snow
    z_slope_ice = True  # Use linear mono slope for autoconversions
    z_slope_liq = True  # Use linear mono slope for autoconversions
    tice = 273.16  # set tice = 165. to turn off ice - phase phys (kessler emulator)
    alin = 842.0  # "a" in lin1983
    clin = 4.8  # "c" in lin 1983, 4.8 -- > 6. (to ehance ql -- > qs)

    @classmethod
    def as_dict(cls):
        return {
            name: default
            for name, default in cls.__dict__.items()
            if not name.startswith("_")
        }


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
                    raise ValueError(
                        "Cannot flatten this namelist, duplicate keys: " + subkey
                    )
                flatter_namelist[subkey] = subvalue
        else:
            flatter_namelist[key] = value
    return flatter_namelist
