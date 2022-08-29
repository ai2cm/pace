from dataclasses import InitVar, dataclass, field, fields
from typing import List, Optional

import gt4py.gtscript as gtscript
import xarray as xr

import pace.dsl.gt4py_utils as gt_utils
import pace.util
from pace.dsl.typing import FloatField
from pace.physics.stencils.microphysics import MicrophysicsState


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
            self.microphysics: Optional[MicrophysicsState] = MicrophysicsState(
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
        else:
            self.microphysics = None

    @classmethod
    def init_zeros(cls, quantity_factory, active_packages: List[str]) -> "PhysicsState":
        initial_storages = {}
        for _field in fields(cls):
            initial_storages[_field.name] = quantity_factory.zeros(
                [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
                _field.metadata["units"],
                dtype=float,
            ).storage
        return cls(
            **initial_storages,
            quantity_factory=quantity_factory,
            active_packages=active_packages,
        )

    @property
    def xr_dataset(self):
        data_vars = {}
        for name, field_info in self.__dataclass_fields__.items():
            if isinstance(field_info.type, gtscript._FieldDescriptor):
                dims = [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM]
                data_vars[name] = xr.DataArray(
                    gt_utils.asarray(getattr(self, name).data),
                    dims=dims,
                    attrs={
                        "long_name": field_info.metadata["name"],
                        "units": field_info.metadata.get("units", "unknown"),
                    },
                )
        return xr.Dataset(data_vars=data_vars)
