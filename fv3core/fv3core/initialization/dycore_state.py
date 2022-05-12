from dataclasses import dataclass, field, fields
from typing import Any, Mapping

import xarray as xr

from pace import util


@dataclass()
class DycoreState:
    u: util.Quantity = field(
        metadata={
            "name": "x_wind",
            "dims": [util.X_DIM, util.Y_INTERFACE_DIM, util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    v: util.Quantity = field(
        metadata={
            "name": "y_wind",
            "dims": [util.X_INTERFACE_DIM, util.Y_DIM, util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    w: util.Quantity = field(
        metadata={
            "name": "vertical_wind",
            "dims": [util.X_DIM, util.Y_DIM, util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    ua: util.Quantity = field(
        metadata={
            "name": "eastward_wind",
            "dims": [util.X_DIM, util.Y_DIM, util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    va: util.Quantity = field(
        metadata={
            "name": "northward_wind",
            "dims": [util.X_DIM, util.Y_DIM, util.Z_DIM],
            "units": "m/s",
        }
    )
    uc: util.Quantity = field(
        metadata={
            "name": "x_wind_on_c_grid",
            "dims": [util.X_INTERFACE_DIM, util.Y_DIM, util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    vc: util.Quantity = field(
        metadata={
            "name": "y_wind_on_c_grid",
            "dims": [util.X_DIM, util.Y_INTERFACE_DIM, util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    delp: util.Quantity = field(
        metadata={
            "name": "pressure_thickness_of_atmospheric_layer",
            "dims": [util.X_DIM, util.Y_DIM, util.Z_DIM],
            "units": "Pa",
            "intent": "inout",
        }
    )
    delz: util.Quantity = field(
        metadata={
            "name": "vertical_thickness_of_atmospheric_layer",
            "dims": [util.X_DIM, util.Y_DIM, util.Z_DIM],
            "units": "m",
            "intent": "inout",
        }
    )
    ps: util.Quantity = field(
        metadata={
            "name": "surface_pressure",
            "dims": [util.X_DIM, util.Y_DIM],
            "units": "Pa",
            "intent": "inout",
        }
    )
    pe: util.Quantity = field(
        metadata={
            "name": "interface_pressure",
            "dims": [util.X_DIM, util.Y_DIM, util.Z_INTERFACE_DIM],
            "units": "Pa",
            "n_halo": 1,
            "intent": "inout",
        }
    )
    pt: util.Quantity = field(
        metadata={
            "name": "air_temperature",
            "dims": [util.X_DIM, util.Y_DIM, util.Z_DIM],
            "units": "degK",
            "intent": "inout",
        }
    )
    peln: util.Quantity = field(
        metadata={
            "name": "logarithm_of_interface_pressure",
            "dims": [
                util.X_DIM,
                util.Y_DIM,
                util.Z_INTERFACE_DIM,
            ],
            "units": "ln(Pa)",
            "n_halo": 0,
            "intent": "inout",
        }
    )
    pk: util.Quantity = field(
        metadata={
            "name": "interface_pressure_raised_to_power_of_kappa",
            "dims": [util.X_DIM, util.Y_DIM, util.Z_INTERFACE_DIM],
            "units": "unknown",
            "n_halo": 0,
            "intent": "inout",
        }
    )
    pkz: util.Quantity = field(
        metadata={
            "name": "layer_mean_pressure_raised_to_power_of_kappa",
            "dims": [util.X_DIM, util.Y_DIM, util.Z_DIM],
            "units": "unknown",
            "n_halo": 0,
            "intent": "inout",
        }
    )
    qvapor: util.Quantity = field(
        metadata={
            "name": "specific_humidity",
            "dims": [util.X_DIM, util.Y_DIM, util.Z_DIM],
            "units": "kg/kg",
        }
    )
    qliquid: util.Quantity = field(
        metadata={
            "name": "cloud_water_mixing_ratio",
            "dims": [util.X_DIM, util.Y_DIM, util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qice: util.Quantity = field(
        metadata={
            "name": "cloud_ice_mixing_ratio",
            "dims": [util.X_DIM, util.Y_DIM, util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qrain: util.Quantity = field(
        metadata={
            "name": "rain_mixing_ratio",
            "dims": [util.X_DIM, util.Y_DIM, util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qsnow: util.Quantity = field(
        metadata={
            "name": "snow_mixing_ratio",
            "dims": [util.X_DIM, util.Y_DIM, util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qgraupel: util.Quantity = field(
        metadata={
            "name": "graupel_mixing_ratio",
            "dims": [util.X_DIM, util.Y_DIM, util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qo3mr: util.Quantity = field(
        metadata={
            "name": "ozone_mixing_ratio",
            "dims": [util.X_DIM, util.Y_DIM, util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qsgs_tke: util.Quantity = field(
        metadata={
            "name": "turbulent_kinetic_energy",
            "dims": [util.X_DIM, util.Y_DIM, util.Z_DIM],
            "units": "m**2/s**2",
            "intent": "inout",
        }
    )
    qcld: util.Quantity = field(
        metadata={
            "name": "cloud_fraction",
            "dims": [util.X_DIM, util.Y_DIM, util.Z_DIM],
            "units": "",
            "intent": "inout",
        }
    )
    q_con: util.Quantity = field(
        metadata={
            "name": "total_condensate_mixing_ratio",
            "dims": [util.X_DIM, util.Y_DIM, util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    omga: util.Quantity = field(
        metadata={
            "name": "vertical_pressure_velocity",
            "dims": [util.X_DIM, util.Y_DIM, util.Z_DIM],
            "units": "Pa/s",
            "intent": "inout",
        }
    )
    mfxd: util.Quantity = field(
        metadata={
            "name": "accumulated_x_mass_flux",
            "dims": [util.X_INTERFACE_DIM, util.Y_DIM, util.Z_DIM],
            "units": "unknown",
            "n_halo": 0,
            "intent": "inout",
        }
    )
    mfyd: util.Quantity = field(
        metadata={
            "name": "accumulated_y_mass_flux",
            "dims": [util.X_DIM, util.Y_INTERFACE_DIM, util.Z_DIM],
            "units": "unknown",
            "n_halo": 0,
            "intent": "inout",
        }
    )
    cxd: util.Quantity = field(
        metadata={
            "name": "accumulated_x_courant_number",
            "dims": [util.X_INTERFACE_DIM, util.Y_DIM, util.Z_DIM],
            "units": "",
            "n_halo": (0, 3),
            "intent": "inout",
        }
    )
    cyd: util.Quantity = field(
        metadata={
            "name": "accumulated_y_courant_number",
            "dims": [util.X_DIM, util.Y_INTERFACE_DIM, util.Z_DIM],
            "units": "",
            "n_halo": (3, 0),
            "intent": "inout",
        }
    )
    diss_estd: util.Quantity = field(
        metadata={
            "name": "dissipation_estimate_from_heat_source",
            "dims": [util.X_DIM, util.Y_DIM, util.Z_DIM],
            "units": "unknown",
            "n_halo": (3, 3),
            "intent": "inout",
        }
    )
    phis: util.Quantity = field(
        metadata={
            "name": "surface_geopotential",
            "units": "m^2 s^-2",
            "dims": [util.X_DIM, util.Y_DIM],
            "intent": "in",
        }
    )
    do_adiabatic_init: bool = field(default=False)
    bdt: float = field(default=0.0)
    mdt: float = field(default=0.0)

    def __post_init__(self):
        for _field in fields(self):
            for check_name in ["units", "dims"]:
                if check_name in _field.metadata:
                    required = _field.metadata[check_name]
                    actual = getattr(getattr(self, _field.name), check_name)
                    if isinstance(required, list):
                        actual = list(actual)
                    if actual != required:
                        raise TypeError(
                            f"{_field.name} has metadata {check_name} of {actual}"
                            f"that does not match the requirement {required}"
                        )

    @classmethod
    def init_zeros(cls, quantity_factory: util.QuantityFactory):
        initial_storages = {}
        for _field in fields(cls):
            if "dims" in _field.metadata.keys():
                initial_storages[_field.name] = quantity_factory.zeros(
                    _field.metadata["dims"], _field.metadata["units"], dtype=float
                ).storage
        return cls.init_from_storages(
            storages=initial_storages, sizer=quantity_factory.sizer
        )

    @classmethod
    def init_from_numpy_arrays(
        cls, dict_of_numpy_arrays, sizer: util.GridSizer, backend: str
    ):
        field_names = [_field.name for _field in fields(cls)]
        for variable_name in dict_of_numpy_arrays.keys():
            if variable_name not in field_names:
                raise KeyError(
                    variable_name + " is provided, but not part of the dycore state"
                )
        dict_state = {}
        for _field in fields(cls):
            if "dims" in _field.metadata.keys():
                dims = _field.metadata["dims"]
                dict_state[_field.name] = util.Quantity(
                    dict_of_numpy_arrays[_field.name],
                    dims,
                    _field.metadata["units"],
                    origin=sizer.get_origin(dims),
                    extent=sizer.get_extent(dims),
                    gt4py_backend=backend,
                )
        state = cls(**dict_state)
        return state

    @classmethod
    def init_from_storages(
        cls,
        storages: Mapping[str, Any],
        sizer: util.GridSizer,
        do_adiabatic_init: bool = False,
        bdt: float = 0.0,
        mdt: float = 0.0,
    ):
        inputs = {}
        for _field in fields(cls):
            if "dims" in _field.metadata.keys():
                dims = _field.metadata["dims"]
                quantity = util.Quantity(
                    storages[_field.name],
                    dims,
                    _field.metadata["units"],
                    origin=sizer.get_origin(dims),
                    extent=sizer.get_extent(dims),
                )
                inputs[_field.name] = quantity
        return cls(**inputs, do_adiabatic_init=do_adiabatic_init, bdt=bdt, mdt=mdt)

    @property
    def xr_dataset(self):
        data_vars = {}
        for name, field_info in self.__dataclass_fields__.items():
            if issubclass(field_info.type, util.Quantity):
                dims = [
                    f"{dim_name}_{name}" for dim_name in field_info.metadata["dims"]
                ]
                data_vars[name] = xr.DataArray(
                    getattr(self, name).data,
                    dims=dims,
                    attrs={
                        "long_name": field_info.metadata["name"],
                        "units": field_info.metadata.get("units", "unknown"),
                    },
                )
        return xr.Dataset(data_vars=data_vars)

    def __getitem__(self, item):
        return getattr(self, item)
