import dataclasses
from fv3core.utils.typing import FloatField, FloatFieldIJ
import copy

from fv3gfs.physics.stencils.microphysics import MicrophysicsState


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
        # [TODO] we may want to use a copy from dycore instead of using the same storages
        return cls(
            qvapor=state.qvapor,
            qliquid=state.qliquid,
            qrain=state.qrain,
            qsnow=state.qsnow,
            qice=state.qice,
            qgraupel=state.qgraupel,
            qo3mr=state.qo3mr,
            qsgs_tke=state.qsgs_tke,
            qcld=state.qcld,
            pt=state.pt,
            delp=state.delp,
            delz=state.delz,
            ua=state.ua,
            va=state.va,
            w=state.w,
            omga=state.omga,
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

    def microphysics(self, storage) -> MicrophysicsState:
        """
        storage: storage for variables not in PhysicsState
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
            storage=storage,
        )
