import pace.util as fv3util
from pace.stencils.c2l_ord import CubedToLatLon
from pace.stencils.testing import ParallelTranslate2Py


class TranslateCubedToLatLon(ParallelTranslate2Py):
    inputs = {
        "u": {
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "v": {
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
    }

    def __init__(self, grids, namelist, stencil_factory):
        super().__init__(grids, namelist, stencil_factory)
        grid = grids[0]
        self.grid_data = grid.grid_data
        self.quantity_factory = grid.quantity_factory
        self.stencil_factory = stencil_factory
        self.grid_indexing = grid.grid_indexing
        self.namelist = namelist
        self._base.in_vars["data_vars"] = {"u": {}, "v": {}, "ua": {}, "va": {}}
        self._base.out_vars = {
            "ua": {},
            "va": {},
            "u": self.grid.y3d_domain_dict(),
            "v": self.grid.x3d_domain_dict(),
        }
        self.stencil_factory = stencil_factory

    def compute_parallel(self, inputs, communicator):
        self._base.make_storage_data_input_vars(inputs)
        u_quantity = _quantity_wrap(
            inputs["u"],
            self.inputs["u"]["dims"],
            self.grid_indexing,
        )
        v_quantity = _quantity_wrap(
            inputs["v"],
            self.inputs["v"]["dims"],
            self.grid_indexing,
        )
        state_dict = {"u": u_quantity, "v": v_quantity}

        self._cubed_to_latlon = CubedToLatLon(
            state_dict,
            self.stencil_factory,
            self.grid_data,
            self.namelist.c2l_ord,
            communicator,
        )
        self._cubed_to_latlon(**inputs)
        return self._base.slice_output(inputs)


def _quantity_wrap(storage, dims, grid_indexing):
    origin, extent = grid_indexing.get_origin_domain(dims)
    return fv3util.Quantity(
        storage,
        dims=dims,
        units="unknown",
        origin=origin,
        extent=extent,
    )
