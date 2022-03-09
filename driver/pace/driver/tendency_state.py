from dataclasses import dataclass, field, fields

from pace import util


@dataclass()
class TendencyState:
    u_dt: util.Quantity = field(
        metadata={
            "name": "eastward_wind_tendency_due_to_physics",
            "dims": [util.X_DIM, util.Y_DIM, util.Z_DIM],
            "units": "m/s**2",
            "intent": "inout",
        }
    )
    v_dt: util.Quantity = field(
        metadata={
            "name": "northward_wind_tendency_due_to_physics",
            "dims": [util.X_DIM, util.Y_DIM, util.Z_DIM],
            "units": "m/s**2",
            "intent": "inout",
        }
    )
    pt_dt: util.Quantity = field(
        metadata={
            "name": "temperature_tendency_due_to_physics",
            "dims": [util.X_DIM, util.Y_DIM, util.Z_DIM],
            "units": "K/s",
            "intent": "inout",
        }
    )

    @classmethod
    def init_zeros(cls, quantity_factory: util.QuantityFactory) -> "TendencyState":
        initial_quantities = {}
        for _field in fields(cls):
            initial_quantities[_field.name] = quantity_factory.zeros(
                _field.metadata["dims"],
                _field.metadata["units"],
                dtype=float,
            )
        return cls(**initial_quantities)
