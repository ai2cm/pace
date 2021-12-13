import copy
import dataclasses

from fv3gfs.physics.stencils.microphysics import MicrophysicsState
from pace.dsl.typing import FloatField


@dataclasses.dataclass()
class PhysicsState:
    """
    Physics state variables
    From dynamical core:
        qvapor: specific_humidity
        qliquid: cloud_water_mixing_ratio
        qrain: rain_mixing_ratio
        qsnow: snow_mixing_ratio
        qice: cloud_ice_mixing_ratio
        qgraupel: graupel_mixing_ratio
        qo3mr: ozone_mixing_ratio
        qsgs_tke: turbulent_kinetic_energy
        qcld: cloud_fraction
        pt: air_temperature
        delp: pressure_thickness_of_atmospheric_layer
        delz: vertical_thickness_of_atmospheric_layer
        ua: eastward_wind
        va: northward_wind
        w: vertical_wind
        omga: vertical_pressure_velocity
    Physics driver:
        delprsi: model_level_pressure_thickness_in_physics
        phii: interface_geopotential_height
        phil: layer_geopotential_height
        dz: geopotential_height_thickness
        *_t1: respective dynamical core variables marched by 1 time step
    Microphysics:
        wmp: layer_mean_vertical_velocity_microph
    """

    qvapor: FloatField
    qliquid: FloatField
    qrain: FloatField
    qsnow: FloatField
    qice: FloatField
    qgraupel: FloatField
    qo3mr: FloatField
    qsgs_tke: FloatField
    qcld: FloatField
    pt: FloatField
    delp: FloatField
    delz: FloatField
    ua: FloatField
    va: FloatField
    w: FloatField
    omga: FloatField
    delprsi: FloatField
    phii: FloatField
    phil: FloatField
    dz: FloatField
    wmp: FloatField
    qvapor_t1: FloatField
    qliquid_t1: FloatField
    qrain_t1: FloatField
    qsnow_t1: FloatField
    qice_t1: FloatField
    qgraupel_t1: FloatField
    qcld_t1: FloatField
    pt_t1: FloatField
    ua_t1: FloatField
    va_t1: FloatField

    @classmethod
    def from_dycore_state(cls, state, storage: FloatField) -> "PhysicsState":
        """
        Constructor for PhysicsState when using dynamical core state
        storage: storage for variables not in dycore
        """
        # [TODO] using a copy here because variables definition change inside physics
        # we should copy only the variables that will be updated
        return cls(
            qvapor=copy.deepcopy(state.qvapor),
            qliquid=copy.deepcopy(state.qliquid),
            qrain=copy.deepcopy(state.qrain),
            qsnow=copy.deepcopy(state.qsnow),
            qice=copy.deepcopy(state.qice),
            qgraupel=copy.deepcopy(state.qgraupel),
            qo3mr=copy.deepcopy(state.qo3mr),
            qsgs_tke=copy.deepcopy(state.qsgs_tke),
            qcld=copy.deepcopy(state.qcld),
            pt=copy.deepcopy(state.pt),
            delp=copy.deepcopy(state.delp),
            delz=copy.deepcopy(state.delz),
            ua=copy.deepcopy(state.ua),
            va=copy.deepcopy(state.va),
            w=copy.deepcopy(state.w),
            omga=copy.deepcopy(state.omga),
            delprsi=copy.deepcopy(storage),
            phii=copy.deepcopy(storage),
            phil=copy.deepcopy(storage),
            dz=copy.deepcopy(storage),
            wmp=copy.deepcopy(storage),
            qvapor_t1=copy.deepcopy(storage),
            qliquid_t1=copy.deepcopy(storage),
            qrain_t1=copy.deepcopy(storage),
            qsnow_t1=copy.deepcopy(storage),
            qice_t1=copy.deepcopy(storage),
            qgraupel_t1=copy.deepcopy(storage),
            qcld_t1=copy.deepcopy(storage),
            pt_t1=copy.deepcopy(storage),
            ua_t1=copy.deepcopy(storage),
            va_t1=copy.deepcopy(storage),
        )

    def microphysics(self, tendency_storage) -> MicrophysicsState:
        """
        tendency_storage: storage for tendency variables not in PhysicsState
        """
        return MicrophysicsState(
            pt=self.pt,
            qvapor=self.qvapor,
            qliquid=self.qliquid,
            qrain=self.qrain,
            qice=self.qice,
            qsnow=self.qsnow,
            qgraupel=self.qgraupel,
            qcld=self.qcld,
            ua=self.ua,
            va=self.va,
            delp=self.delp,
            delz=self.delz,
            omga=self.omga,
            delprsi=self.delprsi,
            wmp=self.wmp,
            dz=self.dz,
            tendency_storage=tendency_storage,
        )
