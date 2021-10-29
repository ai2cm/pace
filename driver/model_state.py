from dataclasses import dataclass, field, fields, InitVar
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK, Float
import copy
import fv3core
from fv3gfs.physics.stencils.microphysics import MicrophysicsState
from stencil_functions import copy_fields_in, fill_gfs, prepare_tendencies_and_update_tracers
from fv_update_phys import ApplyPhysics2Dycore
import fv3gfs.util as fv3util 
from fv3core.decorators import FrozenStencil

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
    ak: FloatFieldK = field(metadata={"name": "atmosphere_hybrid_a_coordinate", "units": "m^2 s^-2", "dims": [fv3util.Z_DIM], "intent":"in"})
    bk: FloatFieldK = field(metadata={"name": "atmosphere_hybrid_b_coordinate", "units": "m^2 s^-2", "dims": [fv3util.Z_DIM], "intent":"in"}) 
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
        
    def __getitem__(self, item):
        return getattr(self, item)

@dataclass()
class PhysicsState:
    qvapor: FloatField = field(metadata={"name": "specific_humidity", "units": "kg/kg", "from_dycore": True})
    qliquid: FloatField = field(metadata={"name": "cloud_water_mixing_ratio", "units": "kg/kg", "intent":"inout", "from_dycore": True})
    qice: FloatField = field(metadata={"name": "cloud_ice_mixing_ratio", "units": "kg/kg", "intent":"inout", "from_dycore": True}) 
    qrain: FloatField = field(metadata={"name": "rain_mixing_ratio", "units": "kg/kg", "intent":"inout", "from_dycore": True})
    qsnow: FloatField = field(metadata={"name": "snow_mixing_ratio","units": "kg/kg", "intent":"inout", "from_dycore": True})
    qgraupel: FloatField = field(metadata={"name": "graupel_mixing_ratio", "units": "kg/kg", "intent":"inout", "from_dycore": True}) 
    qo3mr: FloatField = field(metadata={"name": "ozone_mixing_ratio", "units": "kg/kg", "intent":"inout", "from_dycore": True}) 
    qsgs_tke: FloatField = field(metadata={"name": "turbulent_kinetic_energy","units": "m**2/s**2", "intent":"inout", "from_dycore": True})
    qcld: FloatField = field(metadata={"name": "cloud_fraction","units": "", "intent":"inout", "from_dycore": True})
    pt: FloatField = field(metadata={"name": "air_temperature", "units": "degK", "intent":"inout", "from_dycore": True})
    delp: FloatField = field(metadata={"name": "pressure_thickness_of_atmospheric_layer", "units": "Pa", "intent":"inout", "from_dycore": True})
    delz: FloatField = field(metadata={"name": "vertical_thickness_of_atmospheric_layer", "units": "m", "intent":"inout", "from_dycore": True}) 
    ua: FloatField = field(metadata={"name": "eastward_wind", "units": "m/s", "intent":"inout", "from_dycore": True})
    va: FloatField = field(metadata={"name": "northward_wind", "units": "m/s", "from_dycore": True })
    w: FloatField = field(metadata={"name": "vertical_wind", "units": "m/s", "intent":"inout", "from_dycore": True})
    omga: FloatField = field(metadata={"name": "vertical_pressure_velocity","units": "Pa/s", "intent":"inout", "from_dycore": True})
    qvapor_t1: FloatField = field(metadata={"name": "physics_specific_humidity", "units": "kg/kg", "from_dycore": False})
    qliquid_t1: FloatField = field(metadata={"name": "physics_cloud_water_mixing_ratio", "units": "kg/kg", "intent":"inout", "from_dycore": False})
    qice_t1: FloatField  = field(metadata={"name": "physics_cloud_ice_mixing_ratio", "units": "kg/kg", "intent":"inout", "from_dycore": False}) 
    qrain_t1: FloatField =  field(metadata={"name": "physics_rain_mixing_ratio", "units": "kg/kg", "intent":"inout", "from_dycore": False})
    qsnow_t1: FloatField = field(metadata={"name": "physics_snow_mixing_ratio","units": "kg/kg", "intent":"inout", "from_dycore": False})
    qgraupel_t1: FloatField = field(metadata={"name": "physics_graupel_mixing_ratio", "units": "kg/kg", "intent":"inout", "from_dycore": False}) 
    qcld_t1: FloatField  = field(metadata={"name": "physics_cloud_fraction","units": "", "intent":"inout", "from_dycore": False})
    pt_t1: FloatField  = field(metadata={"name": "physics_air_temperature", "units": "degK", "intent":"inout", "from_dycore": False})
    ua_t1: FloatField = field(metadata={"name": "physics_eastward_wind", "units": "m/s", "intent":"inout", "from_dycore": False})
    va_t1: FloatField  = field(metadata={"name": "physics_northward_wind", "units": "m/s", "from_dycore": False })
    delprsi: FloatField = field(metadata={"name": "model_level_pressure_thickness_in_physics", "units": "Pa", "from_dycore": False})
    phii: FloatField = field(metadata={"name": "interface_geopotential_height", "units": "m", "from_dycore": False})
    phil: FloatField = field(metadata={"name": "layer_geopotential_height", "units": "m", "from_dycore": False})
    dz: FloatField = field(metadata={"name": "geopotential_height_thickness", "units": "m", "from_dycore": False})
    wmp: FloatField = field(metadata={"name": "layer_mean_vertical_velocity_microph", "units": "m/s", "from_dycore": False})
    prsi: FloatField = field(metadata={"name": "interface_pressure", "units": "Pa", "from_dycore": False})
    quantity_factory: InitVar[fv3util.QuantityFactory]
  
    def __post_init__(self, quantity_factory):
        # storage for tendency variables not in PhysicsState
        tendency_storage = quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],  "unknown", dtype=float).storage
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
            tendency_storage=tendency_storage,
        )
        
    @classmethod
    def init_empty(cls, quantity_factory):
        initial_storages = {}
        for field in fields(cls):
            initial_storages[field.name] = quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],  field.metadata["units"], dtype=float).storage
        return cls(**initial_storages, quantity_factory=quantity_factory)
    
    #intended for the case of not copying, reusing the same variables in the dycore and physics
    @classmethod
    def init_from_dycore(cls, quantity_factory, dycore_state):
        initial_storages = {}
        dycore_fields = fields(DycoreState)
        for field in fields(cls):
            if field.metadata["from_dycore"]:
                initial_storages[field.name] = getattr(dycore_state, field.name)
            else:
                initial_storages[field.name] = quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],  field.metadata["units"], dtype=float).storage
        return cls(**initial_storages, quantity_factory=quantity_factory)

class ModelState:
    # TODO, should not need all these inputs
    def __init__(self, dycore_state: DycoreState, physics_state: PhysicsState, quantity_factory, grid, namelist, comm, grid_info):
        self.dycore_state = dycore_state
        self.physics_state = physics_state
        self.state_updater = UpdateAtmosphereState(
            grid, namelist, comm, grid_info, quantity_factory
        )
   
    @classmethod
    def init_empty(cls, grid, quantity_factory,  namelist, comm, grid_info):
         dycore_state = DycoreState.init_empty(quantity_factory)
         physics_state = PhysicsState.init_empty(quantity_factory)
         # If copy gets removed
         # physics_state = PhysicsState.init_from_dycore(quantity_factory, dycore_state)
         return cls(dycore_state, physics_state,  quantity_factory, grid, namelist, comm, grid_info)
    
    @classmethod
    def init_from_serialized_data(cls, serializer, grid, quantity_factory, namelist, comm, grid_info):
        dycore_state = DycoreState.init_from_serialized_data(serializer, grid, quantity_factory)
        physics_state = PhysicsState.init_empty(quantity_factory)
        return cls(dycore_state, physics_state,  quantity_factory, grid, namelist, comm, grid_info)

    @classmethod
    def init_from_dycore_state(cls, dycore_state: DycoreState, grid, quantity_factory: fv3util.QuantityFactory, namelist, comm, grid_info):
        physics_state = PhysicsState.init_empty(quantity_factory)
        # If copy gets removed
        # physics_state = PhysicsState.init_from_dycore(quantity_factory, dycore_state)
        return cls(dycore_state, physics_state,  quantity_factory, grid, namelist, comm, grid_info)
    
    def update_physics_inputs_state(self):
        self.state_updater.copy_from_dycore_to_physics(self.dycore_state, self.physics_state)

    def update_dycore_state(self):
        self.state_updater(self.dycore_state, self.physics_state)


class UpdateAtmosphereState:
    """Fortran name is atmosphere_state_update
    This is an API to apply tendencies and compute a consistent prognostic state.
    """

    def __init__(
            self, grid, namelist, comm: fv3util.CubedSphereCommunicator, grid_info, quantity_factory: fv3util.QuantityFactory
    ):
        self.grid = grid
        self.namelist = namelist
        origin = self.grid.compute_origin()
        shape = self.grid.domain_shape_full(add=(1, 1, 1))
        self._rdt = 1.0 / Float(self.namelist.dt_atmos)
        self._fill_GFS = FrozenStencil(
            fill_gfs,
            origin=self.grid.grid_indexing.origin_full(),
            domain=self.grid.grid_indexing.domain_full(add=(0, 0, 1)),
        )
        self._prepare_tendencies_and_update_tracers = FrozenStencil(
            prepare_tendencies_and_update_tracers,
            origin=self.grid.grid_indexing.origin_compute(),
            domain=self.grid.grid_indexing.domain_compute(add=(0, 0, 1)),
        )
        self._copy_fields_in = FrozenStencil(
            func=copy_fields_in,
            origin=self.grid.grid_indexing.origin_full(),
            domain=self.grid.grid_indexing.domain_full(add=(1, 1, 1))
        )
        self._u_dt = quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],  "m/s^2", dtype=float)
        self._v_dt = quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],  "m/s^2", dtype=float) 
        self._pt_dt = quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],  "degK/s", dtype=float) 
        self._apply_physics2dycore = ApplyPhysics2Dycore(
            self.grid, self.namelist, comm, grid_info
        )
        

    def __call__(
        self,
        dycore_state: DycoreState,
        phy_state: PhysicsState,
    ):
        self._fill_GFS(phy_state.prsi, phy_state.qvapor_t1, 1.0e-9)
        self._prepare_tendencies_and_update_tracers(
            self._u_dt.storage,
            self._v_dt.storage,
            self._pt_dt.storage,
            phy_state.ua_t1,
            phy_state.va_t1,
            phy_state.pt_t1,
            phy_state.qvapor_t1,
            phy_state.qliquid_t1,
            phy_state.qrain_t1,
            phy_state.qsnow_t1,
            phy_state.qice_t1,
            phy_state.qgraupel_t1,
            phy_state.ua,
            phy_state.va,
            phy_state.pt,
            dycore_state.qvapor,
            dycore_state.qliquid,
            dycore_state.qrain,
            dycore_state.qsnow,
            dycore_state.qice,
            dycore_state.qgraupel,
            phy_state.prsi,
            dycore_state.delp,
            self._rdt,
        )
        self._apply_physics2dycore(
            dycore_state, self._u_dt, self._v_dt, self._pt_dt.storage,
        )

    # TODO -- we can probably use the dycore state and not copy
    # check if any of these are modified internally that
    # are fields that are not then overwritten as a last step
    def copy_from_dycore_to_physics(self, dycore_state, physics_state):
        self._copy_fields_in(
            dycore_state.qvapor,
            dycore_state.qliquid,
            dycore_state.qrain,
            dycore_state.qice,
            dycore_state.qsnow,
            dycore_state.qgraupel,
            dycore_state.qo3mr,
            dycore_state.qsgs_tke,
            dycore_state.qcld,
            dycore_state.pt,
            dycore_state.delp,
            dycore_state.delz,
            dycore_state.ua,
            dycore_state.va,
            dycore_state.w,
            dycore_state.omga,
            physics_state.qvapor,
            physics_state.qliquid,
            physics_state.qrain,
            physics_state.qice,
            physics_state.qsnow,
            physics_state.qgraupel,
            physics_state.qo3mr,
            physics_state.qsgs_tke,
            physics_state.qcld,
            physics_state.pt,
            physics_state.delp,
            physics_state.delz,
            physics_state.ua,
            physics_state.va,
            physics_state.w,
            physics_state.omga,
        )
