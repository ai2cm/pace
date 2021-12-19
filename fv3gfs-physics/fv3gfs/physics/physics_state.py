import copy
import dataclasses

from fv3gfs.physics.stencils.microphysics import MicrophysicsState
from pace.dsl.typing import FloatField

from dataclasses import dataclass, field, fields, InitVar
from pace.dsl.typing import FloatField, FloatFieldIJ
from pace.util import Quantity
import copy
import pace.util 
from fv3gfs.physics.stencils.microphysics import MicrophysicsState


@dataclass()
class PhysicsState:
    qvapor: Quantity = field(metadata={"name": "specific_humidity", "units": "kg/kg", "full_model_var": True})
    qliquid: Quantity = field(metadata={"name": "cloud_water_mixing_ratio", "units": "kg/kg", "intent":"inout", "full_model_var": True})
    qice: Quantity = field(metadata={"name": "cloud_ice_mixing_ratio", "units": "kg/kg", "intent":"inout", "full_model_var": True}) 
    qrain: Quantity = field(metadata={"name": "rain_mixing_ratio", "units": "kg/kg", "intent":"inout", "full_model_var": True})
    qsnow: Quantity = field(metadata={"name": "snow_mixing_ratio","units": "kg/kg", "intent":"inout", "full_model_var": True})
    qgraupel: Quantity = field(metadata={"name": "graupel_mixing_ratio", "units": "kg/kg", "intent":"inout", "full_model_var": True}) 
    qo3mr: Quantity = field(metadata={"name": "ozone_mixing_ratio", "units": "kg/kg", "intent":"inout", "full_model_var": True}) 
    qsgs_tke: Quantity = field(metadata={"name": "turbulent_kinetic_energy","units": "m**2/s**2", "intent":"inout", "full_model_var": True})
    qcld: Quantity = field(metadata={"name": "cloud_fraction","units": "", "intent":"inout", "full_model_var": True})
    pt: Quantity = field(metadata={"name": "air_temperature", "units": "degK", "intent":"inout", "full_model_var": True})
    delp: Quantity = field(metadata={"name": "pressure_thickness_of_atmospheric_layer", "units": "Pa", "intent":"inout", "full_model_var": True})
    delz: Quantity = field(metadata={"name": "vertical_thickness_of_atmospheric_layer", "units": "m", "intent":"inout", "full_model_var": True}) 
    ua: Quantity = field(metadata={"name": "eastward_wind", "units": "m/s", "intent":"inout", "full_model_var": True})
    va: Quantity = field(metadata={"name": "northward_wind", "units": "m/s", "full_model_var": True })
    w: Quantity = field(metadata={"name": "vertical_wind", "units": "m/s", "intent":"inout", "full_model_var": True})
    omga: Quantity = field(metadata={"name": "vertical_pressure_velocity","units": "Pa/s", "intent":"inout", "full_model_var": True})
    qvapor_t1: Quantity = field(metadata={"name": "physics_specific_humidity", "units": "kg/kg", "full_model_var": False})
    qliquid_t1: Quantity = field(metadata={"name": "physics_cloud_water_mixing_ratio", "units": "kg/kg", "intent":"inout", "full_model_var": False})
    qice_t1: Quantity  = field(metadata={"name": "physics_cloud_ice_mixing_ratio", "units": "kg/kg", "intent":"inout", "full_model_var": False}) 
    qrain_t1: Quantity =  field(metadata={"name": "physics_rain_mixing_ratio", "units": "kg/kg", "intent":"inout", "full_model_var": False})
    qsnow_t1: Quantity = field(metadata={"name": "physics_snow_mixing_ratio","units": "kg/kg", "intent":"inout", "full_model_var": False})
    qgraupel_t1: Quantity = field(metadata={"name": "physics_graupel_mixing_ratio", "units": "kg/kg", "intent":"inout", "full_model_var": False}) 
    qcld_t1: Quantity  = field(metadata={"name": "physics_cloud_fraction","units": "", "intent":"inout", "full_model_var": False})
    pt_t1: Quantity  = field(metadata={"name": "physics_air_temperature", "units": "degK", "intent":"inout", "full_model_var": False})
    ua_t1: Quantity = field(metadata={"name": "physics_eastward_wind", "units": "m/s", "intent":"inout", "full_model_var": False})
    va_t1: Quantity  = field(metadata={"name": "physics_northward_wind", "units": "m/s", "full_model_var": False })
    delprsi: Quantity = field(metadata={"name": "model_level_pressure_thickness_in_physics", "units": "Pa", "full_model_var": False})
    phii: Quantity = field(metadata={"name": "interface_geopotential_height", "units": "m", "full_model_var": False})
    phil: Quantity = field(metadata={"name": "layer_geopotential_height", "units": "m", "full_model_var": False})
    dz: Quantity = field(metadata={"name": "geopotential_height_thickness", "units": "m", "full_model_var": False})
    wmp: Quantity = field(metadata={"name": "layer_mean_vertical_velocity_microph", "units": "m/s", "full_model_var": False})
    prsi: Quantity = field(metadata={"name": "interface_pressure", "units": "Pa", "full_model_var": False})
    prsik: Quantity = field(metadata={"name": "log interface_pressure", "units": "Pa", "full_model_var": False})
    quantity_factory: InitVar[pace.util.QuantityFactory]
  
    def __post_init__(self, quantity_factory):
        # storage for tendency variables not in PhysicsState
        tendency = quantity_factory.zeros([pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],  "unknown", dtype=float)
        self.microphysics = MicrophysicsState(
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
            tendency_storage=tendency,
        )
        
    @classmethod
    def init_empty(cls, quantity_factory):
        initial_storages = {}
        for field in fields(cls):
            initial_storages[field.name] = quantity_factory.zeros([pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],  field.metadata["units"], dtype=float)
        return cls(**initial_storages, quantity_factory=quantity_factory)
    
   
   
