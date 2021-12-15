from dataclasses import InitVar, dataclass, field, fields
from typing import Optional

import pace.util as fv3util
from pace.dsl.typing import FloatField, FloatFieldIJ


@dataclass()
class DycoreState:
    u: FloatField = field(
        metadata={
            "name": "x_wind",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    v: FloatField = field(
        metadata={
            "name": "y_wind",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    w: FloatField = field(
        metadata={
            "name": "vertical_wind",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    ua: FloatField = field(
        metadata={
            "name": "eastward_wind",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    va: FloatField = field(
        metadata={
            "name": "northward_wind",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        }
    )
    uc: FloatField = field(
        metadata={
            "name": "x_wind_on_c_grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    vc: FloatField = field(
        metadata={
            "name": "y_wind_on_c_grid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    delp: FloatField = field(
        metadata={
            "name": "pressure_thickness_of_atmospheric_layer",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "Pa",
            "intent": "inout",
        }
    )
    delz: FloatField = field(
        metadata={
            "name": "vertical_thickness_of_atmospheric_layer",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m",
            "intent": "inout",
        }
    )
    ps: FloatFieldIJ = field(
        metadata={
            "name": "surface_pressure",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "Pa",
            "intent": "inout",
        }
    )
    pe: FloatField = field(
        metadata={
            "name": "interface_pressure",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_INTERFACE_DIM],
            "units": "Pa",
            "n_halo": 1,
            "intent": "inout",
        }
    )
    pt: FloatField = field(
        metadata={
            "name": "air_temperature",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "degK",
            "intent": "inout",
        }
    )
    peln: FloatField = field(
        metadata={
            "name": "logarithm_of_interface_pressure",
            "dims": [
                fv3util.X_DIM,
                fv3util.Y_DIM,
                fv3util.Z_INTERFACE_DIM,
            ],
            "units": "ln(Pa)",
            "n_halo": 0,
            "intent": "inout",
        }
    )
    pk: FloatField = field(
        metadata={
            "name": "interface_pressure_raised_to_power_of_kappa",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_INTERFACE_DIM],
            "units": "unknown",
            "n_halo": 0,
            "intent": "inout",
        }
    )
    pkz: FloatField = field(
        metadata={
            "name": "layer_mean_pressure_raised_to_power_of_kappa",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "unknown",
            "n_halo": 0,
            "intent": "inout",
        }
    )
    qvapor: FloatField = field(
        metadata={
            "name": "specific_humidity",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
        }
    )
    qliquid: FloatField = field(
        metadata={
            "name": "cloud_water_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qice: FloatField = field(
        metadata={
            "name": "cloud_ice_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qrain: FloatField = field(
        metadata={
            "name": "rain_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qsnow: FloatField = field(
        metadata={
            "name": "snow_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qgraupel: FloatField = field(
        metadata={
            "name": "graupel_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qo3mr: FloatField = field(
        metadata={
            "name": "ozone_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qsgs_tke: FloatField = field(
        metadata={
            "name": "turbulent_kinetic_energy",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m**2/s**2",
            "intent": "inout",
        }
    )
    qcld: FloatField = field(
        metadata={
            "name": "cloud_fraction",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "",
            "intent": "inout",
        }
    )
    q_con: FloatField = field(
        metadata={
            "name": "total_condensate_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    omga: FloatField = field(
        metadata={
            "name": "vertical_pressure_velocity",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "Pa/s",
            "intent": "inout",
        }
    )
    mfxd: FloatField = field(
        metadata={
            "name": "accumulated_x_mass_flux",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "unknown",
            "n_halo": 0,
            "intent": "inout",
        }
    )
    mfyd: FloatField = field(
        metadata={
            "name": "accumulated_y_mass_flux",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "unknown",
            "n_halo": 0,
            "intent": "inout",
        }
    )
    cxd: FloatField = field(
        metadata={
            "name": "accumulated_x_courant_number",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "",
            "n_halo": (0, 3),
            "intent": "inout",
        }
    )
    cyd: FloatField = field(
        metadata={
            "name": "accumulated_y_courant_number",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "",
            "n_halo": (3, 0),
            "intent": "inout",
        }
    )
    diss_estd: FloatField = field(
        metadata={
            "name": "dissipation_estimate_from_heat_source",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "unknown",
            "n_halo": (3, 3),
            "intent": "inout",
        }
    )
    phis: FloatField = field(
        metadata={
            "name": "surface_geopotential",
            "units": "m^2 s^-2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "intent": "in",
        }
    )
    quantity_factory: InitVar[fv3util.QuantityFactory]
    do_adiabatic_init: bool = field(default=False)
    bdt: float = field(default=0.0)
    mdt: float = field(default=0.0)

    def __post_init__(self, quantity_factory: Optional[fv3util.QuantityFactory]):
        if quantity_factory is not None:
            # creating quantities around the storages
            # TODO, when dycore and physics use quantities everywhere
            # change fields to be quantities and remove this extra processing
            for _field in fields(self):
                if "dims" in _field.metadata.keys():
                    dims = _field.metadata["dims"]
                    quantity = fv3util.Quantity(
                        getattr(self, _field.name),
                        dims,
                        _field.metadata["units"],
                        origin=quantity_factory._sizer.get_origin(dims),
                        extent=quantity_factory._sizer.get_extent(dims),
                    )
                    setattr(self, _field.name + "_quantity", quantity)

    @classmethod
    def init_empty(cls, quantity_factory):
        initial_storages = {}
        for _field in fields(cls):
            if "dims" in _field.metadata.keys():
                initial_storages[_field.name] = quantity_factory.zeros(
                    _field.metadata["dims"], _field.metadata["units"], dtype=float
                ).storage
        return cls(**initial_storages, quantity_factory=quantity_factory)

    @classmethod
    def init_from_numpy_arrays(cls, dict_of_numpy_arrays, quantity_factory, backend):
        field_names = [_field.name for _field in fields(cls)]
        for variable_name, data in dict_of_numpy_arrays.items():
            if variable_name not in field_names:
                raise KeyError(
                    variable_name + " is provided, but not part of the dycore state"
                )
        dict_state = {}
        for _field in fields(cls):
            if "dims" in _field.metadata.keys():
                dims = _field.metadata["dims"]
                dict_state[_field.name] = fv3util.Quantity(
                    dict_of_numpy_arrays[_field.name],
                    dims,
                    _field.metadata["units"],
                    origin=quantity_factory._sizer.get_origin(dims),
                    extent=quantity_factory._sizer.get_extent(dims),
                    gt4py_backend=backend,
                ).storage  # TODO remove .storage when using only Quantities
        state = cls(**dict_state, quantity_factory=quantity_factory)
        return state

    @classmethod
    def init_from_quantities(cls, dict_of_quantities):
        field_names = [field.name for field in fields(cls)]
        for variable_name, data in dict_of_quantities.items():
            if variable_name not in field_names:
                raise KeyError(
                    variable_name + " is provided, but not part of the dycore state"
                )
        for field_name in field_names:
            if field_name not in dict_of_quantities.keys():
                raise KeyError(
                    field_name
                    + " is not included in the provided dictionary of quantities"
                )
            elif not isinstance(dict_of_quantities[field_name], fv3util.Quantity):
                raise TypeError(
                    field_name
                    + " is not a Quantity, but instead a "
                    + str(type(dict_of_quantities[field_name]))
                )
        for _field in fields(cls):
            for varcheck in ["units", "dims"]:
                if (
                    _field.metadata[varcheck]
                    != dict_of_quantities[_field.name][varcheck]
                ):
                    raise TypeError(
                        field_name
                        + " has metadata "
                        + varcheck
                        + " of "
                        + dict_of_quantities[_field.name][varcheck]
                        + " that does not match the requirement"
                        + _field.metadata[varcheck]
                    )
        return cls(**dict_of_quantities, quantity_factory=None)

    def __getitem__(self, item):
        return getattr(self, item)
