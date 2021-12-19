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
            tendency_storage=tendency.storage,
        )
        
    @classmethod
    def init_empty(cls, quantity_factory):
        initial_storages = {}
        for field in fields(cls):
            initial_storages[field.name] = quantity_factory.zeros([pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],  field.metadata["units"], dtype=float).storage
        return cls(**initial_storages, quantity_factory=quantity_factory)
    
   
    @classmethod
    def init_from_numpy_arrays(cls, dict_of_numpy_arrays, quantity_factory):
        state = cls.init_empty(quantity_factory)
        field_names = [field.name for field in fields(cls)]
        for variable_name, data in dict_of_numpy_arrays:
            if not variable_name in field_names:
                raise KeyError(variable_name + ' is provided, but not part of the dycore state')
            getattr(state, variable_name).data[:] = data
        for field_name in field_names:
            if not field_name in dict_of_numpy_arrays.keys():
                raise KeyError(field_name + ' is not included in the provided dictionary of numpy arrays')
        return state


    @classmethod
    def init_from_quantities(cls, dict_of_quantities):
        field_names = [field.name for field in fields(cls)]
        for variable_name, data in dict_of_quantities:
            if not variable_name in field_names:
                raise KeyError(variable_name + ' is provided, but not part of the dycore state')
            getattr(state, variable_name).data[:] = data
        for field_name in field_names:
            if not field_name in dict_of_quantities.keys():
                raise KeyError(field_name + ' is not included in the provided dictionary of quantities')
            elif not isinstance(dict_of_quantities[field_name], pace.util.Quantity):
                raise TypeError(field_name + ' is not a Quantity, but instead a ' + type(dict_of_quantities[field_name]))
        return cls(**dict_of_quantities, quantity_factory=None)
"""
@dataclasses.dataclass()
class PhysicsState:
    
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
       
        Constructor for PhysicsState when using dynamical core state
        storage: storage for variables not in dycore
    
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
       
        tendency_storage: storage for tendency variables not in PhysicsState
    
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
"""
