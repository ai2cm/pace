import copy
from dataclasses import InitVar, dataclass, field, fields
from typing import List

import pace.util
from fv3core.initialization.dycore_state import DycoreState
from fv3gfs.physics.stencils.microphysics import MicrophysicsState
from pace.dsl.typing import FloatField


@dataclass()
class PhysicsState:
    qvapor: FloatField = field(metadata={"name": "specific_humidity", "units": "kg/kg"})
    qliquid: FloatField = field(
        metadata={
            "name": "cloud_water_mixing_ratio",
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qice: FloatField = field(
        metadata={"name": "cloud_ice_mixing_ratio", "units": "kg/kg", "intent": "inout"}
    )
    qrain: FloatField = field(
        metadata={"name": "rain_mixing_ratio", "units": "kg/kg", "intent": "inout"}
    )
    qsnow: FloatField = field(
        metadata={"name": "snow_mixing_ratio", "units": "kg/kg", "intent": "inout"}
    )
    qgraupel: FloatField = field(
        metadata={"name": "graupel_mixing_ratio", "units": "kg/kg", "intent": "inout"}
    )
    qo3mr: FloatField = field(
        metadata={"name": "ozone_mixing_ratio", "units": "kg/kg", "intent": "inout"}
    )
    qsgs_tke: FloatField = field(
        metadata={
            "name": "turbulent_kinetic_energy",
            "units": "m**2/s**2",
            "intent": "inout",
        }
    )
    qcld: FloatField = field(
        metadata={"name": "cloud_fraction", "units": "", "intent": "inout"}
    )
    pt: FloatField = field(
        metadata={"name": "air_temperature", "units": "degK", "intent": "inout"}
    )
    delp: FloatField = field(
        metadata={
            "name": "pressure_thickness_of_atmospheric_layer",
            "units": "Pa",
            "intent": "inout",
        }
    )
    delz: FloatField = field(
        metadata={
            "name": "vertical_thickness_of_atmospheric_layer",
            "units": "m",
            "intent": "inout",
        }
    )
    ua: FloatField = field(
        metadata={"name": "eastward_wind", "units": "m/s", "intent": "inout"}
    )
    va: FloatField = field(
        metadata={"name": "northward_wind", "units": "m/s", "intent": "inout"}
    )
    w: FloatField = field(
        metadata={"name": "vertical_wind", "units": "m/s", "intent": "inout"}
    )
    omga: FloatField = field(
        metadata={
            "name": "vertical_pressure_velocity",
            "units": "Pa/s",
            "intent": "inout",
        }
    )
    physics_updated_specific_humidity: FloatField = field(
        metadata={
            "name": "physics_specific_humidity",
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    physics_updated_qliquid: FloatField = field(
        metadata={
            "name": "physics_cloud_water_mixing_ratio",
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    physics_updated_qice: FloatField = field(
        metadata={
            "name": "physics_cloud_ice_mixing_ratio",
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    physics_updated_qrain: FloatField = field(
        metadata={
            "name": "physics_rain_mixing_ratio",
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    physics_updated_qsnow: FloatField = field(
        metadata={
            "name": "physics_snow_mixing_ratio",
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    physics_updated_qgraupel: FloatField = field(
        metadata={
            "name": "physics_graupel_mixing_ratio",
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    physics_updated_cloud_fraction: FloatField = field(
        metadata={"name": "physics_cloud_fraction", "units": "", "intent": "inout"}
    )
    physics_updated_pt: FloatField = field(
        metadata={"name": "physics_air_temperature", "units": "degK", "intent": "inout"}
    )
    physics_updated_ua: FloatField = field(
        metadata={"name": "physics_eastward_wind", "units": "m/s", "intent": "inout"}
    )
    physics_updated_va: FloatField = field(
        metadata={"name": "physics_northward_wind", "units": "m/s", "intent": "inout"}
    )
    delprsi: FloatField = field(
        metadata={
            "name": "model_level_pressure_thickness_in_physics",
            "units": "Pa",
            "intent": "inout",
        }
    )
    phii: FloatField = field(
        metadata={
            "name": "interface_geopotential_height",
            "units": "m",
            "intent": "inout",
        }
    )
    phil: FloatField = field(
        metadata={"name": "layer_geopotential_height", "units": "m", "intent": "inout"}
    )
    dz: FloatField = field(
        metadata={
            "name": "geopotential_height_thickness",
            "units": "m",
            "intent": "inout",
        }
    )
    wmp: FloatField = field(
        metadata={
            "name": "layer_mean_vertical_velocity_microph",
            "units": "m/s",
            "intent": "inout",
        }
    )
    prsi: FloatField = field(
        metadata={"name": "interface_pressure", "units": "Pa", "intent": "inout"}
    )
    prsik: FloatField = field(
        metadata={"name": "log_interface_pressure", "units": "Pa", "intent": "inout"}
    )
    quantity_factory: InitVar[pace.util.QuantityFactory]
    active_packages: InitVar[List[str]]

    def __post_init__(
        self, quantity_factory: pace.util.QuantityFactory, active_packages: List[str]
    ):
        # storage for tendency variables not in PhysicsState
        if "microphysics" in active_packages:
            tendency = quantity_factory.zeros(
                [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
                "unknown",
                dtype=float,
            ).storage
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
    def from_dycore_state(
        cls,
        state: DycoreState,
        quantity_factory: pace.util.QuantityFactory,
        active_packages: List[str],
    ) -> "PhysicsState":
        """
        Constructor for PhysicsState when using dynamical core state

        Args:
            quantity_factory: used to initialize storages not present
                in the dycore state
            active_packages: names of physics packages active
        """
        storage = quantity_factory.zeros(
            [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "unknown",
            dtype=float,
        ).storage
        # [TODO] using a copy here because variables definition change inside physics
        # we should copy only the variables that will be updated
        return cls(
            qvapor=copy.deepcopy(state.qvapor.storage),
            qliquid=copy.deepcopy(state.qliquid.storage),
            qrain=copy.deepcopy(state.qrain.storage),
            qsnow=copy.deepcopy(state.qsnow.storage),
            qice=copy.deepcopy(state.qice.storage),
            qgraupel=copy.deepcopy(state.qgraupel.storage),
            qo3mr=copy.deepcopy(state.qo3mr.storage),
            qsgs_tke=copy.deepcopy(state.qsgs_tke.storage),
            qcld=copy.deepcopy(state.qcld.storage),
            pt=copy.deepcopy(state.pt.storage),
            delp=copy.deepcopy(state.delp.storage),
            delz=copy.deepcopy(state.delz.storage),
            ua=copy.deepcopy(state.ua.storage),
            va=copy.deepcopy(state.va.storage),
            w=copy.deepcopy(state.w.storage),
            omga=copy.deepcopy(state.omga),
            delprsi=copy.deepcopy(storage),
            phii=copy.deepcopy(storage),
            phil=copy.deepcopy(storage),
            dz=copy.deepcopy(storage),
            wmp=copy.deepcopy(storage),
            prsi=copy.deepcopy(storage),
            prsik=copy.deepcopy(storage),
            physics_updated_specific_humidity=copy.deepcopy(storage),
            physics_updated_qliquid=copy.deepcopy(storage),
            physics_updated_qrain=copy.deepcopy(storage),
            physics_updated_qsnow=copy.deepcopy(storage),
            physics_updated_qice=copy.deepcopy(storage),
            physics_updated_qgraupel=copy.deepcopy(storage),
            physics_updated_cloud_fraction=copy.deepcopy(storage),
            physics_updated_pt=copy.deepcopy(storage),
            physics_updated_ua=copy.deepcopy(storage),
            physics_updated_va=copy.deepcopy(storage),
            quantity_factory=quantity_factory,
            active_packages=active_packages,
        )

    @classmethod
    def init_zeros(cls, quantity_factory) -> "PhysicsState":
        initial_storages = {}
        for _field in fields(cls):
            initial_storages[_field.name] = quantity_factory.zeros(
                [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
                _field.metadata["units"],
                dtype=float,
            ).storage
        return cls(**initial_storages, quantity_factory=quantity_factory)
