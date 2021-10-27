from dataclasses import dataclass, field, fields, InitVar
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK
import copy
import fv3core
from fv3gfs.physics.stencils.microphysics import MicrophysicsState
import fv3gfs.util as fv3util 


@dataclass()
class DycoreState:
    u: FloatField = field(metadata={"name": "x_wind", "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM], "units": "m/s", "intent":"inout"})
    v: FloatField = field(metadata={"name": "y_wind", "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],"units": "m/s", "intent":"inout"})
    w: FloatField = field(metadata={"name": "vertical_wind", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "m/s", "intent":"inout"})
    ua: FloatField = field(metadata={"name": "eastward_wind", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],"units": "m/s", "intent":"inout"})
    va: FloatField = field(metadata={"name": "northward_wind", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],"units": "m/s", })
    uc: FloatField = field(metadata={"name": "x_wind_on_c_grid","dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],"units": "m/s", "intent":"inout"})
    vc: FloatField = field(metadata={"name": "y_wind_on_c_grid", "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM], "units": "m/s", "intent":"inout"})
    delp: FloatField = field(metadata={"name": "pressure_thickness_of_atmospheric_layer", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "Pa", "intent":"inout"})
    delz: FloatField = field(metadata={"name": "vertical_thickness_of_atmospheric_layer", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "m", "intent":"inout"}) 
    ps: FloatFieldIJ = field(metadata={"name": "surface_pressure","dims": [fv3util.X_DIM, fv3util.Y_DIM], "units": "Pa", "intent":"inout"})
    pe: FloatField = field(metadata={"name": "interface_pressure", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_INTERFACE_DIM], "units": "Pa","n_halo": 1, "intent":"inout"}) 
    pt: FloatField = field(metadata={"name": "air_temperature", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "degK", "intent":"inout"})
    peln: FloatField = field(metadata={"name": "logarithm_of_interface_pressure", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_INTERFACE_DIM,], "units": "ln(Pa)", "n_halo": 0, "intent":"inout"}) 
    pk: FloatField = field(metadata={"name": "interface_pressure_raised_to_power_of_kappa", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_INTERFACE_DIM], "units": "unknown", "n_halo": 0, "intent":"inout"})
    pkz: FloatField = field(metadata={"name": "layer_mean_pressure_raised_to_power_of_kappa", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],"units": "unknown", "n_halo": 0, "intent":"inout"})
    qvapor: FloatField = field(metadata={"name": "specific_humidity", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "kg/kg"})
    qliquid: FloatField = field(metadata={"name": "cloud_water_mixing_ratio", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "kg/kg", "intent":"inout"})
    qice: FloatField = field(metadata={"name": "cloud_ice_mixing_ratio", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "kg/kg", "intent":"inout"}) 
    qrain: FloatField = field(metadata={"name": "rain_mixing_ratio", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "kg/kg", "intent":"inout"})
    qsnow: FloatField = field(metadata={"name": "snow_mixing_ratio", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],"units": "kg/kg", "intent":"inout"})
    qgraupel: FloatField = field(metadata={"name": "graupel_mixing_ratio", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "kg/kg", "intent":"inout"}) 
    qo3mr: FloatField = field(metadata={"name": "ozone_mixing_ratio", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "kg/kg", "intent":"inout"}) 
    qsgs_tke: FloatField = field(metadata={"name": "turbulent_kinetic_energy", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],"units": "m**2/s**2", "intent":"inout"})
    qcld: FloatField = field(metadata={"name": "cloud_fraction", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],"units": "", "intent":"inout"})
    q_con: FloatField = field(metadata={"name": "total_condensate_mixing_ratio", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],"units": "kg/kg", "intent":"inout"})
    omga: FloatField = field(metadata={"name": "vertical_pressure_velocity","dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],"units": "Pa/s", "intent":"inout"})
    mfxd: FloatField = field(metadata={"name": "accumulated_x_mass_flux", "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "unknown", "n_halo": 0, "intent":"inout"}) 
    mfyd: FloatField = field(metadata={"name": "accumulated_y_mass_flux","dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM], "units": "unknown", "n_halo": 0, "intent":"inout"}) 
    cxd: FloatField = field(metadata={"name": "accumulated_x_courant_number", "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "","n_halo": (0, 3), "intent":"inout"}) 
    cyd: FloatField = field(metadata={"name": "accumulated_y_courant_number", "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM], "units": "", "n_halo": (3, 0), "intent":"inout"}) 
    diss_estd: FloatField = field(metadata={"name": "dissipation_estimate_from_heat_source", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "unknown", "intent":"inout"})
    phis: FloatField = field(metadata={"name": "surface_geopotential", "units": "m^2 s^-2", "dims": [fv3util.X_DIM, fv3util.Y_DIM], "intent":"in"})
    ak: FloatField = field(metadata={"name": "atmosphere_hybrid_a_coordinate", "units": "m^2 s^-2", "dims": [fv3util.Z_DIM], "intent":"in"})
    bk: FloatField = field(metadata={"name": "atmosphere_hybrid_b_coordinate", "units": "m^2 s^-2", "dims": [fv3util.Z_DIM], "intent":"in"}) 
    quantity_factory: InitVar[fv3util.QuantityFactory]
    do_adiabatic_init: bool = field(default=False)
    bdt: float = field(default=0.0)
    mdt: float = field(default=0.0)

    def __post_init__(self, quantity_factory):
        # creating quantities around the storages
        # TODO, when dycore and physics use quantities everywhere
        # change fields to be quantities and remove this extra processing
        for field in fields(self):
            if "dims" in field.metadata.keys():
                dims = field.metadata["dims"]
                quantity = fv3util.Quantity(
                    getattr(self, field.name),
                    dims,
                    field.metadata["units"],
                    origin=quantity_factory._sizer.get_origin(dims),
                    extent=quantity_factory._sizer.get_extent(dims),
                )
                setattr(self, field.name + '_quantity', quantity)
    @classmethod
    def init_empty(cls, quantity_factory):
        initial_storages = {}
        for field in fields(cls):
            if "dims" in field.metadata.keys():
                initial_storages[field.name] = quantity_factory.zeros(field.metadata["dims"], field.metadata["units"], dtype=float).storage
                print('what', field.name,  initial_storages[field.name].shape)
        return cls(**initial_storages, quantity_factory=quantity_factory)
    
    def __getitem__(self, item):
        return getattr(self, item)

@dataclass()
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
#def storage_factory():

#def quantity_factory():
# InitVar -- only used in initialization
#  condition: InitVar[str] = None
@dataclass()
class ModelState:
    """
    dynamical_core_state
    physics_state
    """

    u: FloatField = field(metadata={"name": "x_wind", "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM], "units": "m/s", "intent":"inout"})
    v: FloatField = field(metadata={"name": "y_wind", "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],"units": "m/s", "intent":"inout"})
    w: FloatField = field(metadata={"name": "vertical_wind", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "m/s", "intent":"inout"})
    ua: FloatField = field(metadata={"name": "eastward_wind", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],"units": "m/s", "intent":"inout"})
    va: FloatField = field(metadata={"name": "northward_wind", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],"units": "m/s", })
    uc: FloatField = field(metadata={"name": "x_wind_on_c_grid","dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],"units": "m/s", "intent":"inout"})
    vc: FloatField = field(metadata={"name": "y_wind_on_c_grid", "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM], "units": "m/s", "intent":"inout"})
    delp: FloatField = field(metadata={"name": "pressure_thickness_of_atmospheric_layer", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "Pa", "intent":"inout"})
    delz: FloatField = field(metadata={"name": "vertical_thickness_of_atmospheric_layer", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "m", "intent":"inout"}) 
    ps: FloatFieldIJ = field(metadata={"name": "surface_pressure","dims": [fv3util.X_DIM, fv3util.Y_DIM], "units": "Pa", "intent":"inout"})
    pe: FloatField = field(metadata={"name": "interface_pressure", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_INTERFACE_DIM], "units": "Pa","n_halo": 1, "intent":"inout"}) 
    pt: FloatField = field(metadata={"name": "air_temperature", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "degK", "intent":"inout"})
    peln: FloatField = field(metadata={"name": "logarithm_of_interface_pressure", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_INTERFACE_DIM,], "units": "ln(Pa)", "n_halo": 0, "intent":"inout"}) 
    pk: FloatField = field(metadata={"name": "interface_pressure_raised_to_power_of_kappa", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_INTERFACE_DIM], "units": "unknown", "n_halo": 0, "intent":"inout"})
    pkz: FloatField = field(metadata={"name": "layer_mean_pressure_raised_to_power_of_kappa", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],"units": "unknown", "n_halo": 0, "intent":"inout"})
    qvapor: FloatField = field(metadata={"name": "specific_humidity", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "kg/kg"})
    qliquid: FloatField = field(metadata={"name": "cloud_water_mixing_ratio", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "kg/kg", "intent":"inout"})
    qice: FloatField = field(metadata={"name": "cloud_ice_mixing_ratio", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "kg/kg", "intent":"inout"}) 
    qrain: FloatField = field(metadata={"name": "rain_mixing_ratio", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "kg/kg", "intent":"inout"})
    qsnow: FloatField = field(metadata={"name": "snow_mixing_ratio", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],"units": "kg/kg", "intent":"inout"})
    qgraupel: FloatField = field(metadata={"name": "graupel_mixing_ratio", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "kg/kg", "intent":"inout"}) 
    qo3mr: FloatField = field(metadata={"name": "ozone_mixing_ratio", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "kg/kg", "intent":"inout"}) 
    qsgs_tke: FloatField = field(metadata={"name": "turbulent_kinetic_energy", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],"units": "m**2/s**2", "intent":"inout"})
    qcld: FloatField = field(metadata={"name": "cloud_fraction", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],"units": "", "intent":"inout"})
    q_con: FloatField = field(metadata={"name": "total_condensate_mixing_ratio", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],"units": "kg/kg", "intent":"inout"})
    omga: FloatField = field(metadata={"name": "vertical_pressure_velocity","dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],"units": "Pa/s", "intent":"inout"})
    mfxd: FloatField = field(metadata={"name": "accumulated_x_mass_flux", "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "unknown", "n_halo": 0, "intent":"inout"}) 
    mfyd: FloatField = field(metadata={"name": "accumulated_y_mass_flux","dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM], "units": "unknown", "n_halo": 0, "intent":"inout"}) 
    cxd: FloatField = field(metadata={"name": "accumulated_x_courant_number", "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "","n_halo": (0, 3), "intent":"inout"}) 
    cyd: FloatField = field(metadata={"name": "accumulated_y_courant_number", "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM], "units": "", "n_halo": (3, 0), "intent":"inout"}) 
    diss_estd: FloatField = field(metadata={"name": "dissipation_estimate_from_heat_source", "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "unknown", "intent":"inout"})
    phis: FloatField = field(metadata={"name": "surface_geopotential", "units": "m^2 s^-2", "dims": [fv3util.X_DIM, fv3util.Y_DIM], "intent":"in"})
    ak: FloatField = field(metadata={"name": "atmosphere_hybrid_a_coordinate", "units": "m^2 s^-2", "dims": [fv3util.Z_DIM], "intent":"in"})
    bk: FloatField = field(metadata={"name": "atmosphere_hybrid_b_coordinate", "units": "m^2 s^-2", "dims": [fv3util.Z_DIM], "intent":"in"}) 
    quantity_factory: InitVar[fv3util.QuantityFactory]
    do_adiabatic_init: bool = field(default=False)
    bdt: float = field(default=0.0)
    mdt: float = field(default=0.0)
    
    
    # glue
    #u_dt: FloatField
    #v_dt: FloatField
    #u_quantity: Quantity = field(default_factory=quantity_factory, metadata={"name": "x_wind", "units": "m/s", "intent": "inout"})
    
    #@classmethod
    #def initialize_empty(cls, quantity_factory):
    #    field_types = {field.name: field.type for field in fields(cls)}
    def dycore_state(self) -> DycoreState:
        return Dycorestate()

    def physics_state(self) ->PhysicsState:
        return PhysicsState()

    def __post_init__(self, quantity_factory):
        # creating quantities around the storages
        # TODO, when dycore and physics use quantities everywhere
        # change fields to be quantities and remove this extra processing
        for field in fields(self):
            if "dims" in field.metadata.keys():
                dims = field.metadata["dims"]
                quantity = fv3util.Quantity(
                    getattr(self, field.name),
                    dims,
                    field.metadata["units"],
                    origin=quantity_factory._sizer.get_origin(dims),
                    extent=quantity_factory._sizer.get_extent(dims),
                )
                setattr(self, field.name + '_quantity', quantity)
    @classmethod
    def init_empty(cls, quantity_factory):
        initial_storages = {}
        for field in fields(cls):
            if "dims" in field.metadata.keys():
                initial_storages[field.name] = quantity_factory.zeros(field.metadata["dims"], field.metadata["units"], dtype=float).storage
                print('what', field.name,  initial_storages[field.name].shape)
        return cls(**initial_storages, quantity_factory=quantity_factory)
    
            
    @classmethod
    def init_from_serialized_data(cls, serializer, grid, quantity_factory):
        savepoint_in = serializer.get_savepoint("FVDynamics-In")[0]
        driver_object = fv3core.testing.TranslateFVDynamics([grid])
        input_data = driver_object.collect_input_data(serializer, savepoint_in)
        # making just storages for the moment, revisit if making them all quantities (maybe use state_from_inputs)
        driver_object._base.make_storage_data_input_vars(input_data)
        # used for the translate test as inputs, but are generated by the MetricsTerms class and are not part of this data class
        for delvar in ["ptop", "ks"]:
            del input_data[delvar]
        return cls(**input_data, quantity_factory=quantity_factory)
"""
    @classmethod
    def init_baroclinic(cls, quantity_factory, test_case=13):
        numpy_state = baroclinic_initialization.compute()
        make_quantities
        make_storages
        #self._quantity_factory.zeros(
        #    [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM], "radians", dtype=float
        #)
        #fv3util.Quantity(
        #            input_data,
        #            dims,
        #            properties["units"],
        #            origin=grid.sizer.get_origin(dims),
        #            extent=grid.sizer.get_extent(dims),
        #        )
        return cls(**input_data, quantity_factory=quantity_factory)
"""
        
"""
from dataclasses import dataclass, InitVar, field

@dataclass
class A:
    temp: InitVar[str]
    a: int = field(init=False)
    def __post_init__(self, temp):
        self.a = int(temp)
"""
