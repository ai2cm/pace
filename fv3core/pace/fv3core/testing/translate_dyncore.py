import pace.dsl
import pace.dsl.gt4py_utils as utils
import pace.fv3core.stencils.dyn_core as dyn_core
import pace.util
from pace.fv3core import DycoreState, DynamicalCoreConfig
from pace.stencils.testing import ParallelTranslate2PyState


class TranslateDynCore(ParallelTranslate2PyState):
    inputs = {
        "q_con": {
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "default",
        },
        "cappa": {
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "default",
        },
        "delp": {
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "default",
        },
        "pt": {
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "K",
        },
        "u": {
            "dims": [pace.util.X_DIM, pace.util.Y_INTERFACE_DIM, pace.util.Z_DIM],
            "units": "m/s",
        },
        "v": {
            "dims": [pace.util.X_INTERFACE_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m/s",
        },
        "uc": {
            "dims": [pace.util.X_INTERFACE_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m/s",
        },
        "vc": {
            "dims": [pace.util.X_DIM, pace.util.Y_INTERFACE_DIM, pace.util.Z_DIM],
            "units": "m/s",
        },
        "w": {
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m/s",
        },
    }

    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
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

        self._base.in_vars["parameters"] = ["mdt", "akap", "ptop", "n_map"]

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
        self.ignore_near_zero_errors["wsd"] = 1e-18
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
        self._base.make_storage_data_input_vars(inputs)
        state = DycoreState.init_zeros(quantity_factory=self.grid.quantity_factory)
        wsd: pace.util.Quantity = self.grid.quantity_factory.empty(
            dims=[pace.util.X_DIM, pace.util.Y_DIM],
            units="unknown",
        )
        for name, value in inputs.items():
            if hasattr(state, name) and isinstance(state[name], pace.util.Quantity):
                # storage can have buffer points at the end, so value.shape
                # is often not equal to state[name].storage.shape
                selection = tuple(slice(0, end) for end in value.shape)
                state[name].storage[selection] = value
            else:
                setattr(state, name, value)
        acoustic_dynamics = dyn_core.AcousticDynamics(
            comm=communicator,
            stencil_factory=self.stencil_factory,
            grid_data=grid_data,
            damping_coefficients=self.grid.damping_coefficients,
            grid_type=self.grid.grid_type,
            nested=self.grid.nested,
            stretched_grid=self.grid.stretched_grid,
            config=DynamicalCoreConfig.from_namelist(self.namelist).acoustic_dynamics,
            pfull=inputs["pfull"],
            phis=inputs["phis"],
            wsd=wsd.storage,
            state=state,
        )
        acoustic_dynamics.cappa.storage[:] = inputs["cappa"][:]

        acoustic_dynamics(state, timestep=inputs["mdt"], n_map=state.n_map)
        # the "inputs" dict is not used to return, we construct a new dict based
        # on variables attached to `state`
        storages_only = {}
        for name, value in vars(state).items():
            if isinstance(value, pace.util.Quantity):
                storages_only[name] = value.storage
            else:
                storages_only[name] = value
        storages_only["wsd"] = wsd.storage
        storages_only["cappa"] = acoustic_dynamics.cappa.storage
        return self._base.slice_output(storages_only)
