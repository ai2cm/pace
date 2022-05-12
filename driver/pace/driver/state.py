import dataclasses

import xarray as xr

import fv3core
import fv3gfs.physics
import pace.util
import pace.util.grid


@dataclasses.dataclass()
class TendencyState:
    """
    Accumulated tendencies from physical parameterizations to be applied
    to the dynamical core model state.
    """

    u_dt: pace.util.Quantity = dataclasses.field(
        metadata={
            "name": "eastward_wind_tendency_due_to_physics",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m/s**2",
            "intent": "inout",
        }
    )
    v_dt: pace.util.Quantity = dataclasses.field(
        metadata={
            "name": "northward_wind_tendency_due_to_physics",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m/s**2",
            "intent": "inout",
        }
    )
    pt_dt: pace.util.Quantity = dataclasses.field(
        metadata={
            "name": "temperature_tendency_due_to_physics",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "K/s",
            "intent": "inout",
        }
    )

    @classmethod
    def init_zeros(cls, quantity_factory: pace.util.QuantityFactory) -> "TendencyState":
        initial_quantities = {}
        for _field in dataclasses.fields(cls):
            initial_quantities[_field.name] = quantity_factory.zeros(
                _field.metadata["dims"],
                _field.metadata["units"],
                dtype=float,
            )
        return cls(**initial_quantities)

    @property
    def xr_dataset(self):
        data_vars = {}
        for name, field_info in self.__dataclass_fields__.items():
            if issubclass(field_info.type, pace.util.Quantity):
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


@dataclasses.dataclass
class DriverState:
    dycore_state: fv3core.DycoreState
    physics_state: fv3gfs.physics.PhysicsState
    tendency_state: TendencyState
    grid_data: pace.util.grid.GridData
    damping_coefficients: pace.util.grid.DampingCoefficients
    driver_grid_data: pace.util.grid.DriverGridData
