from dataclasses import InitVar, dataclass, field, fields
from typing import Any, Dict, List, Mapping, Optional

import xarray as xr

import pace.dsl.gt4py_utils as gt_utils
import pace.util
from pace.physics.stencils.microphysics import MicrophysicsState


@dataclass()
class PhysicsState:
    qvapor: pace.util.Quantity = field(
        metadata={
            "name": "specific_humidity",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
        }
    )
    qliquid: pace.util.Quantity = field(
        metadata={
            "name": "cloud_water_mixing_ratio",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qice: pace.util.Quantity = field(
        metadata={
            "name": "cloud_ice_mixing_ratio",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qrain: pace.util.Quantity = field(
        metadata={
            "name": "rain_mixing_ratio",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qsnow: pace.util.Quantity = field(
        metadata={
            "name": "snow_mixing_ratio",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qgraupel: pace.util.Quantity = field(
        metadata={
            "name": "graupel_mixing_ratio",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qo3mr: pace.util.Quantity = field(
        metadata={
            "name": "ozone_mixing_ratio",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qsgs_tke: pace.util.Quantity = field(
        metadata={
            "name": "turbulent_kinetic_energy",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m**2/s**2",
            "intent": "inout",
        }
    )
    qcld: pace.util.Quantity = field(
        metadata={
            "name": "cloud_fraction",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "",
            "intent": "inout",
        }
    )
    pt: pace.util.Quantity = field(
        metadata={
            "name": "air_temperature",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "degK",
            "intent": "inout",
        }
    )
    delp: pace.util.Quantity = field(
        metadata={
            "name": "pressure_thickness_of_atmospheric_layer",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "Pa",
            "intent": "inout",
        }
    )
    delz: pace.util.Quantity = field(
        metadata={
            "name": "vertical_thickness_of_atmospheric_layer",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m",
            "intent": "inout",
        }
    )
    ua: pace.util.Quantity = field(
        metadata={
            "name": "eastward_wind",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    va: pace.util.Quantity = field(
        metadata={
            "name": "northward_wind",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m/s",
        }
    )
    w: pace.util.Quantity = field(
        metadata={
            "name": "vertical_wind",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    omga: pace.util.Quantity = field(
        metadata={
            "name": "vertical_pressure_velocity",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "Pa/s",
            "intent": "inout",
        }
    )
    physics_updated_specific_humidity: pace.util.Quantity = field(
        metadata={
            "name": "physics_updated_specific_humidity",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
        }
    )
    physics_updated_qliquid: pace.util.Quantity = field(
        metadata={
            "name": "physics_updated_liquid_water_mixing_ratio",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    physics_updated_qice: pace.util.Quantity = field(
        metadata={
            "name": "physics_updated_ice_water_mixing_ratio",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    physics_updated_qrain: pace.util.Quantity = field(
        metadata={
            "name": "physics_updated_rain_water_mixing_ratio",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    physics_updated_qsnow: pace.util.Quantity = field(
        metadata={
            "name": "physics_updated_snow_mixing_ratio",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    physics_updated_qgraupel: pace.util.Quantity = field(
        metadata={
            "name": "physics_updated_graupel_mixing_ratio",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    physics_updated_cloud_fraction: pace.util.Quantity = field(
        metadata={
            "name": "physics_cloud_fraction",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "",
            "intent": "inout",
        }
    )
    physics_updated_pt: pace.util.Quantity = field(
        metadata={
            "name": "physics_air_temperature",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "degK",
            "intent": "inout",
        }
    )
    physics_updated_ua: pace.util.Quantity = field(
        metadata={
            "name": "physics_eastward_wind",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    physics_updated_va: pace.util.Quantity = field(
        metadata={
            "name": "physics_northward_wind",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    delprsi: pace.util.Quantity = field(
        metadata={
            "name": "model_level_pressure_thickness_in_physics",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "Pa",
            "intent": "inout",
        }
    )
    phii: pace.util.Quantity = field(
        metadata={
            "name": "interface_geopotential_height",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_INTERFACE_DIM],
            "units": "m",
            "intent": "inout",
        }
    )
    phil: pace.util.Quantity = field(
        metadata={
            "name": "layer_geopotential_height",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m",
            "intent": "inout",
        }
    )
    dz: pace.util.Quantity = field(
        metadata={
            "name": "geopotential_height_thickness",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m",
            "intent": "inout",
        }
    )
    wmp: pace.util.Quantity = field(
        metadata={
            "name": "layer_mean_vertical_velocity_microph",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    prsi: pace.util.Quantity = field(
        metadata={
            "name": "interface_pressure",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_INTERFACE_DIM],
            "units": "Pa",
            "intent": "inout",
        }
    )
    prsik: pace.util.Quantity = field(
        metadata={
            "name": "log_interface_pressure",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_INTERFACE_DIM],
            "units": "Pa",
            "intent": "inout",
        }
    )
    land: pace.util.Quantity = field(
        metadata={
            "name": "land_mask",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM],
            "units": "-",
            "intent": "in",
        }
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
            )
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
                tendency=tendency,
                land=self.land,
            )
        else:
            self.microphysics = None

    @classmethod
    def init_zeros(cls, quantity_factory, active_packages: List[str]) -> "PhysicsState":
        initial_arrays = {}
        for _field in fields(cls):
            if "dims" in _field.metadata.keys():
                initial_arrays[_field.name] = quantity_factory.zeros(
                    _field.metadata["dims"], _field.metadata["units"], dtype=float
                ).data
        return cls(
            **initial_arrays,
            quantity_factory=quantity_factory,
            active_packages=active_packages,
        )

    @classmethod
    def init_from_storages(
        cls,
        storages: Mapping[str, Any],
        sizer: pace.util.GridSizer,
        quantity_factory: pace.util.QuantityFactory,
        active_packages: List[str],
    ) -> "PhysicsState":
        inputs: Dict[str, pace.util.Quantity] = {}
        for _field in fields(cls):
            if "dims" in _field.metadata.keys():
                dims = _field.metadata["dims"]
                quantity = pace.util.Quantity(
                    storages[_field.name],
                    dims,
                    _field.metadata["units"],
                    origin=sizer.get_origin(dims),
                    extent=sizer.get_extent(dims),
                )
                inputs[_field.name] = quantity
        return cls(
            **inputs, quantity_factory=quantity_factory, active_packages=active_packages
        )

    @property
    def xr_dataset(self):
        data_vars = {}
        for name, field_info in self.__dataclass_fields__.items():
            if name not in ["quantity_factory", "active_packages"]:
                if issubclass(field_info.type, pace.util.Quantity):
                    dims = [
                        f"{dim_name}_{name}" for dim_name in field_info.metadata["dims"]
                    ]
                    data_vars[name] = xr.DataArray(
                        gt_utils.asarray(getattr(self, name).data),
                        dims=dims,
                        attrs={
                            "long_name": field_info.metadata["name"],
                            "units": field_info.metadata.get("units", "unknown"),
                        },
                    )
        return xr.Dataset(data_vars=data_vars)
