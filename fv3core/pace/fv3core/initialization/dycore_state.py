from dataclasses import dataclass, field, fields
from typing import Any, Mapping

import xarray as xr

import pace.dsl.gt4py_utils as gt_utils
import pace.util


@dataclass()
class DycoreState:
    u: pace.util.Quantity = field(
        metadata={
            "name": "x_wind",
            "dims": [pace.util.X_DIM, pace.util.Y_INTERFACE_DIM, pace.util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    v: pace.util.Quantity = field(
        metadata={
            "name": "y_wind",
            "dims": [pace.util.X_INTERFACE_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
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
    # TODO: move a-grid winds to temporary internal storage
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
    uc: pace.util.Quantity = field(
        metadata={
            "name": "x_wind_on_c_grid",
            "dims": [pace.util.X_INTERFACE_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    vc: pace.util.Quantity = field(
        metadata={
            "name": "y_wind_on_c_grid",
            "dims": [pace.util.X_DIM, pace.util.Y_INTERFACE_DIM, pace.util.Z_DIM],
            "units": "m/s",
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
    ps: pace.util.Quantity = field(
        metadata={
            "name": "surface_pressure",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM],
            "units": "Pa",
            "intent": "inout",
        }
    )
    pe: pace.util.Quantity = field(
        metadata={
            "name": "interface_pressure",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_INTERFACE_DIM],
            "units": "Pa",
            "n_halo": 1,
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
    peln: pace.util.Quantity = field(
        metadata={
            "name": "logarithm_of_interface_pressure",
            "dims": [
                pace.util.X_DIM,
                pace.util.Y_DIM,
                pace.util.Z_INTERFACE_DIM,
            ],
            "units": "ln(Pa)",
            "n_halo": 0,
            "intent": "inout",
        }
    )
    pk: pace.util.Quantity = field(
        metadata={
            "name": "interface_pressure_raised_to_power_of_kappa",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_INTERFACE_DIM],
            "units": "unknown",
            "n_halo": 0,
            "intent": "inout",
        }
    )
    pkz: pace.util.Quantity = field(
        metadata={
            "name": "layer_mean_pressure_raised_to_power_of_kappa",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "n_halo": 0,
            "intent": "inout",
        }
    )
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
    q_con: pace.util.Quantity = field(
        metadata={
            "name": "total_condensate_mixing_ratio",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
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
    mfxd: pace.util.Quantity = field(
        metadata={
            "name": "accumulated_x_mass_flux",
            "dims": [pace.util.X_INTERFACE_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "n_halo": 0,
            "intent": "inout",
        }
    )
    mfyd: pace.util.Quantity = field(
        metadata={
            "name": "accumulated_y_mass_flux",
            "dims": [pace.util.X_DIM, pace.util.Y_INTERFACE_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "n_halo": 0,
            "intent": "inout",
        }
    )
    cxd: pace.util.Quantity = field(
        metadata={
            "name": "accumulated_x_courant_number",
            "dims": [pace.util.X_INTERFACE_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "",
            "n_halo": (0, 3),
            "intent": "inout",
        }
    )
    cyd: pace.util.Quantity = field(
        metadata={
            "name": "accumulated_y_courant_number",
            "dims": [pace.util.X_DIM, pace.util.Y_INTERFACE_DIM, pace.util.Z_DIM],
            "units": "",
            "n_halo": (3, 0),
            "intent": "inout",
        }
    )
    diss_estd: pace.util.Quantity = field(
        metadata={
            "name": "dissipation_estimate_from_heat_source",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "n_halo": (3, 3),
            "intent": "inout",
        }
    )
    """
    how much energy is dissipated, is mainly captured
    to send to the stochastic physics (in contrast to heat_source)
    """
    phis: pace.util.Quantity = field(
        metadata={
            "name": "surface_geopotential",
            "units": "m^2 s^-2",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM],
            "intent": "in",
        }
    )
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
    def init_zeros(cls, quantity_factory: pace.util.QuantityFactory):
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
        cls, dict_of_numpy_arrays, sizer: pace.util.GridSizer, backend: str
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
                dict_state[_field.name] = pace.util.Quantity(
                    dict_of_numpy_arrays[_field.name],
                    dims,
                    _field.metadata["units"],
                    origin=sizer.get_origin(dims),
                    extent=sizer.get_extent(dims),
                    gt4py_backend=backend,
                )
        state = cls(**dict_state)  # type: ignore
        return state

    @classmethod
    def init_from_storages(
        cls,
        storages: Mapping[str, Any],
        sizer: pace.util.GridSizer,
        bdt: float = 0.0,
        mdt: float = 0.0,
    ):
        inputs = {}
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
        return cls(**inputs, bdt=bdt, mdt=mdt)

    @classmethod
    def from_fortran_restart(
        cls,
        *,
        quantity_factory: pace.util.QuantityFactory,
        communicator: pace.util.CubedSphereCommunicator,
        path: str,
    ):
        state_dict: Mapping[str, pace.util.Quantity] = pace.util.open_restart(
            dirname=path,
            communicator=communicator,
            tracer_properties=TRACER_PROPERTIES,
        )

        new = cls.init_zeros(quantity_factory=quantity_factory)
        new.pt.view[:] = new.pt.np.asarray(
            state_dict["air_temperature"].transpose(new.pt.dims).view[:]
        )
        new.delp.view[:] = new.delp.np.asarray(
            state_dict["pressure_thickness_of_atmospheric_layer"]
            .transpose(new.delp.dims)
            .view[:]
        )
        new.phis.view[:] = new.phis.np.asarray(
            state_dict["surface_geopotential"].transpose(new.phis.dims).view[:]
        )
        new.w.view[:] = new.w.np.asarray(
            state_dict["vertical_wind"].transpose(new.w.dims).view[:]
        )
        new.u.view[:] = new.u.np.asarray(
            state_dict["x_wind"].transpose(new.u.dims).view[:]
        )
        new.v.view[:] = new.v.np.asarray(
            state_dict["y_wind"].transpose(new.v.dims).view[:]
        )
        new.qvapor.view[:] = new.qvapor.np.asarray(
            state_dict["specific_humidity"].transpose(new.qvapor.dims).view[:]
        )
        new.qliquid.view[:] = new.qliquid.np.asarray(
            state_dict["cloud_liquid_water_mixing_ratio"]
            .transpose(new.qliquid.dims)
            .view[:]
        )
        new.qice.view[:] = new.qice.np.asarray(
            state_dict["cloud_ice_mixing_ratio"].transpose(new.qice.dims).view[:]
        )
        new.qrain.view[:] = new.qrain.np.asarray(
            state_dict["rain_mixing_ratio"].transpose(new.qrain.dims).view[:]
        )
        new.qsnow.view[:] = new.qsnow.np.asarray(
            state_dict["snow_mixing_ratio"].transpose(new.qsnow.dims).view[:]
        )
        new.qgraupel.view[:] = new.qgraupel.np.asarray(
            state_dict["graupel_mixing_ratio"].transpose(new.qgraupel.dims).view[:]
        )
        new.qo3mr.view[:] = new.qo3mr.np.asarray(
            state_dict["ozone_mixing_ratio"].transpose(new.qo3mr.dims).view[:]
        )
        new.qcld.view[:] = new.qcld.np.asarray(
            state_dict["cloud_fraction"].transpose(new.qcld.dims).view[:]
        )
        new.delz.view[:] = new.delz.np.asarray(
            state_dict["vertical_thickness_of_atmospheric_layer"]
            .transpose(new.delz.dims)
            .view[:]
        )

        return new

    @property
    def xr_dataset(self):
        data_vars = {}
        for name, field_info in self.__dataclass_fields__.items():
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

    def __getitem__(self, item):
        return getattr(self, item)


TRACER_PROPERTIES = {
    "specific_humidity": {
        "dims": [pace.util.Z_DIM, pace.util.Y_DIM, pace.util.X_DIM],
        "restart_name": "sphum",
        "units": "g/kg",
    },
    "cloud_liquid_water_mixing_ratio": {
        "dims": [pace.util.Z_DIM, pace.util.Y_DIM, pace.util.X_DIM],
        "restart_name": "liq_wat",
        "units": "g/kg",
    },
    "cloud_ice_mixing_ratio": {
        "dims": [pace.util.Z_DIM, pace.util.Y_DIM, pace.util.X_DIM],
        "restart_name": "ice_wat",
        "units": "g/kg",
    },
    "rain_mixing_ratio": {
        "dims": [pace.util.Z_DIM, pace.util.Y_DIM, pace.util.X_DIM],
        "restart_name": "rainwat",
        "units": "g/kg",
    },
    "snow_mixing_ratio": {
        "dims": [pace.util.Z_DIM, pace.util.Y_DIM, pace.util.X_DIM],
        "restart_name": "snowwat",
        "units": "g/kg",
    },
    "graupel_mixing_ratio": {
        "dims": [pace.util.Z_DIM, pace.util.Y_DIM, pace.util.X_DIM],
        "restart_name": "graupel",
        "units": "g/kg",
    },
    "ozone_mixing_ratio": {
        "dims": [pace.util.Z_DIM, pace.util.Y_DIM, pace.util.X_DIM],
        "restart_name": "o3mr",
        "units": "g/kg",
    },
    "turbulent_kinetic_energy": {
        "dims": [pace.util.Z_DIM, pace.util.Y_DIM, pace.util.X_DIM],
        "restart_name": "sgs_tke",
        "units": "g/kg",
    },
    "cloud_fraction": {
        "dims": [pace.util.Z_DIM, pace.util.Y_DIM, pace.util.X_DIM],
        "restart_name": "cld_amt",
        "units": "g/kg",
    },
}
