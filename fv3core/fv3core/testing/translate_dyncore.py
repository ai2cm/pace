import fv3core.stencils.dyn_core as dyn_core
import pace.dsl.gt4py_utils as utils
import pace.util as fv3util
from fv3core.initialization.dycore_state import DycoreState
from pace.stencils.testing import ParallelTranslate2PyState


class TranslateDynCore(ParallelTranslate2PyState):
    inputs = {
        "q_con": {
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "default",
        },
        "cappa": {
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "default",
        },
        "delp": {
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "default",
        },
        "pt": {"dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "K"},
        "u": {
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "v": {
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "uc": {
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "vc": {
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "w": {"dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "m/s"},
    }

    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self._base.in_vars["data_vars"] = {
            "cappa": {},
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "w": {},
            "delz": {},
            "delp": {},
            "pt": {},
            "pe": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
                "kaxis": 1,
            },
            "pk": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
            },
            "phis": {"kstart": 0, "kend": 0},
            "wsd": grid.compute_dict(),
            "omga": {},
            "ua": {},
            "va": {},
            "uc": grid.x3d_domain_dict(),
            "vc": grid.y3d_domain_dict(),
            "mfxd": grid.x3d_compute_dict(),
            "mfyd": grid.y3d_compute_dict(),
            "cxd": grid.x3d_compute_domain_y_dict(),
            "cyd": grid.y3d_compute_domain_x_dict(),
            "pkz": grid.compute_dict(),
            "peln": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
                "kaxis": 1,
            },
            "q_con": {},
            "ak": {},
            "bk": {},
            "diss_estd": {},
            "pfull": {},
        }
        self._base.in_vars["data_vars"]["wsd"]["kstart"] = grid.npz
        self._base.in_vars["data_vars"]["wsd"]["kend"] = None

        self._base.in_vars["parameters"] = [
            "mdt",
            "akap",
            "ptop",
            "n_map",
            "ks",
        ]

        self._base.out_vars = {}
        for v, d in self._base.in_vars["data_vars"].items():
            self._base.out_vars[v] = d

        del self._base.out_vars["ak"]
        del self._base.out_vars["bk"]
        del self._base.out_vars["pfull"]
        del self._base.out_vars["phis"]

        # TODO: Fix edge_interpolate4 in d2a2c_vect to match closer and the
        # variables here should as well.
        self.max_error = 2e-6
        self.stencil_factory = stencil_factory
        self.namelist = namelist

    def compute_parallel(self, inputs, communicator):
        # ak, bk, pfull, and phis are numpy arrays at this point and
        #   must be converted into gt4py storages
        for name in ("ak", "bk", "pfull", "phis"):
            inputs[name] = utils.make_storage_data(
                inputs[name],
                inputs[name].shape,
                len(inputs[name].shape) * (0,),
                backend=self.stencil_factory.backend,
            )

        grid_data = self.grid.grid_data
        if grid_data.ak is None or grid_data.bk is None:
            grid_data.ak = inputs["ak"]
            grid_data.bk = inputs["bk"]
            grid_data.ptop = inputs["ptop"]
            grid_data.ks = inputs["ks"]
        acoustic_dynamics = dyn_core.AcousticDynamics(
            communicator,
            self.stencil_factory,
            grid_data,
            self.grid.damping_coefficients,
            self.grid.grid_type,
            self.grid.nested,
            self.grid.stretched_grid,
            self.namelist.acoustic_dynamics,
            inputs["pfull"],
            inputs["phis"],
        )
        self._base.make_storage_data_input_vars(inputs)
        state = DycoreState.init_zeros(quantity_factory=self.grid.quantity_factory)
        state.cappa = self.grid.quantity_factory.empty(
            dims=[fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            units="unknown",
        )
        for name, value in inputs.items():
            if hasattr(state, name) and isinstance(state[name], fv3util.Quantity):
                # storage can have buffer points at the end, so value.shape
                # is often not equal to state[name].storage.shape
                selection = tuple(slice(0, end) for end in value.shape)
                state[name].storage[selection] = value
            else:
                setattr(state, name, value)
        acoustic_dynamics(state)
        storages_only = {}
        for name, value in vars(state).items():
            if isinstance(value, fv3util.Quantity):
                storages_only[name] = value.storage
            else:
                storages_only[name] = value
        return self._base.slice_output(storages_only)
