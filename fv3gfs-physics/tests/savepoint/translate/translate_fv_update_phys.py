import dataclasses

import numpy as np

import pace.dsl.gt4py_utils as utils
import pace.util
from pace.dsl.typing import FloatField, FloatFieldIJ
from pace.stencils.fv_update_phys import ApplyPhysicsToDycore
from pace.stencils.testing.translate_physics import ParallelPhysicsTranslate2Py


@dataclasses.dataclass()
class DycoreState:
    u: FloatField
    v: FloatField
    delp: FloatField
    ps: FloatFieldIJ
    pe: FloatField
    pt: FloatField
    peln: FloatField
    pk: FloatField
    qvapor: FloatField
    qliquid: FloatField
    qice: FloatField
    qrain: FloatField
    qsnow: FloatField
    qgraupel: FloatField
    ua: FloatField
    va: FloatField

    def __getitem__(self, item):
        return getattr(self, item)


class TranslateFVUpdatePhys(ParallelPhysicsTranslate2Py):
    def __init__(self, grids, namelist, stencil_factory):
        super().__init__(grids, namelist, stencil_factory)
        grid = grids[0]
        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing
        self._base.in_vars["data_vars"] = {
            "u_dt": {},
            "v_dt": {},
            "t_dt": {},
            "ua": {},
            "va": {},
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "peln": {
                "istart": self.grid_indexing.isc,
                "iend": self.grid_indexing.iec,
                "jstart": self.grid_indexing.jsc,
                "jend": self.grid_indexing.jec,
                "kend": namelist.npz,
                "kaxis": 1,
            },
            "delp": {},
            "pt": {},
            "ps": {},
            "pe": {
                "istart": self.grid_indexing.isc - 1,
                "iend": self.grid_indexing.iec + 1,
                "jstart": self.grid_indexing.jsc - 1,
                "jend": self.grid_indexing.jec + 1,
                "kend": namelist.npz + 1,
                "kaxis": 1,
            },
            "pk": grid.compute_buffer_k_dict(),
        }
        self._base.out_vars = {
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "pt": {},
            "u": {},
            "v": {},
            "ua": {},
            "va": {},
        }
        self.namelist = namelist

    def collect_input_data(self, serializer, savepoint):
        input_data = {}
        for varname in [*self._base.in_vars["data_vars"]]:
            info = self._base.in_vars["data_vars"][varname]
            dwind_format = info["dwind"] if "dwind" in info else False
            if "serialname" in info:
                serialname = info["serialname"]
            else:
                serialname = varname
            if dwind_format:
                dwind_data_dict = self.read_dwind_serialized_data(
                    serializer, savepoint, serialname
                )
                for dvar in dwind_data_dict.keys():
                    input_data[dvar] = dwind_data_dict[dvar]
            else:
                input_data[serialname] = self.read_dycore_serialized_data(
                    serializer, savepoint, serialname
                )
        return input_data

    def read_dycore_serialized_data(self, serializer, savepoint, variable):
        data = serializer.read(variable, savepoint)
        if len(data.flatten()) == 1:
            return data[0]
        return data

    def read_dwind_serialized_data(self, serializer, savepoint, varname):
        max_shape = self.grid.domain_shape_full(add=(1, 1, 1))
        start_indices = {
            "vlon": (self.grid.isd + 1, self.grid.jsd + 1),
            "vlat": (self.grid.isd + 1, self.grid.jsd + 1),
        }
        axes = {"edge_vect_s": 0, "edge_vect_n": 0, "edge_vect_w": 1, "edge_vect_e": 1}
        input_data = {}
        data = serializer.read(varname, savepoint)

        # convert single element numpy arrays to scalars
        if data.size == 1:
            data = data.item()
            input_data[varname] = data
            return input_data
        elif len(data.shape) < 2:
            start1 = start_indices.get(varname, 0)
            size1 = data.shape[0]
            axis = axes.get(varname, 2)
            input_data[varname] = np.zeros(max_shape[axis])
            input_data[varname][start1 : start1 + size1] = data
        elif len(data.shape) == 2:
            input_data[varname] = np.zeros(max_shape[0:2])
            start1, start2 = start_indices.get(varname, (0, 0))
            size1, size2 = data.shape
            input_data[varname][start1 : start1 + size1, start2 : start2 + size2] = data
        else:
            start1, start2, start3 = start_indices.get(varname, self.grid.full_origin())
            size1, size2, size3 = data.shape
            input_data[varname] = np.zeros(max_shape)
            input_data[varname][
                start1 : start1 + size1,
                start2 : start2 + size2,
                start3 : start3 + size3,
            ] = data
        input_data[varname] = utils.make_storage_data(
            data=input_data[varname],
            origin=self.grid_indexing.origin_full(),
            shape=input_data[varname].shape,
            backend=self.stencil_factory.backend,
        )
        return input_data

    def storage_vars(self):
        return self._base.in_vars["data_vars"]

    def make_storage_data_input_vars(self, inputs, storage_vars=None):
        if storage_vars is None:
            storage_vars = self.storage_vars()
        for p in self._base.in_vars["parameters"]:
            if type(inputs[p]) in [np.int64, np.int32]:
                inputs[p] = int(inputs[p])
        for d, info in storage_vars.items():
            serialname = info["serialname"] if "serialname" in info else d
            self._base.update_info(info, inputs)
            if "kaxis" in info:
                inputs[serialname] = np.moveaxis(inputs[serialname], info["kaxis"], 2)
            istart, jstart, kstart = self._base.collect_start_indices(
                inputs[serialname].shape, info
            )

            names_4d = None
            if len(inputs[serialname].shape) == 4:
                names_4d = info.get("names_4d", utils.tracer_variables)

            dummy_axes = info.get("dummy_axes", None)
            axis = info.get("axis", 2)
            inputs[d] = self._base.make_storage_data(
                np.squeeze(inputs[serialname]),
                istart=istart,
                jstart=jstart,
                kstart=kstart,
                dummy_axes=dummy_axes,
                axis=axis,
                names_4d=names_4d,
                read_only=d not in self._base.write_vars,
                full_shape="full_shape" in storage_vars[d],
            )
            if d != serialname:
                del inputs[serialname]

    def compute_parallel(self, inputs, communicator):
        self.make_storage_data_input_vars(inputs)

        tendencies = {}
        for key in ["u_dt", "v_dt", "t_dt"]:
            storage = inputs.pop(key)
            tendencies[key] = pace.util.Quantity(
                storage,
                dims=[pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
                units="test",
                origin=(0, 0, 0),
                extent=storage.shape,
            )
        partitioner = pace.util.CubedSpherePartitioner(
            pace.util.TilePartitioner(self.namelist.layout)
        )
        self._base.compute_func = ApplyPhysicsToDycore(
            self.stencil_factory,
            self.grid.grid_data,
            self.namelist,
            communicator,
            self.grid.driver_grid_data,
        )
        state = DycoreState(**inputs)
        dims_u = [pace.util.X_DIM, pace.util.Y_INTERFACE_DIM, pace.util.Z_DIM]
        u_quantity = self.grid.make_quantity(
            state.u,
            dims=dims_u,
            origin=self.grid.sizer.get_origin(dims_u),
            extent=self.grid.sizer.get_extent(dims_u),
        )
        dims_v = [pace.util.X_INTERFACE_DIM, pace.util.Y_DIM, pace.util.Z_DIM]
        v_quantity = self.grid.make_quantity(
            state.v,
            dims=dims_v,
            origin=self.grid.sizer.get_origin(dims_v),
            extent=self.grid.sizer.get_extent(dims_v),
        )
        state.u = u_quantity
        state.v = v_quantity
        self._base.compute_func(
            state,
            tendencies["u_dt"],
            tendencies["v_dt"],
            tendencies["t_dt"],
            dt=float(self.namelist.dt_atmos),
        )
        out = {}
        ds = self.grid.default_domain_dict()
        out["qvapor"] = state.qvapor[self.grid.slice_dict(ds)]
        out["qliquid"] = state.qliquid[self.grid.slice_dict(ds)]
        out["qice"] = state.qice[self.grid.slice_dict(ds)]
        out["qrain"] = state.qrain[self.grid.slice_dict(ds)]
        out["qsnow"] = state.qsnow[self.grid.slice_dict(ds)]
        out["qgraupel"] = state.qgraupel[self.grid.slice_dict(ds)]
        out["pt"] = state.pt[self.grid.slice_dict(ds)]
        state.u.storage.synchronize()
        state.v.storage.synchronize()
        state.ua.synchronize()
        state.va.synchronize()
        out["u"] = np.asarray(state.u.storage)[self.grid.y3d_domain_interface()]
        out["v"] = np.asarray(state.v.storage)[self.grid.x3d_domain_interface()]
        out["ua"] = state.ua[self.grid.slice_dict(ds)]
        out["va"] = state.va[self.grid.slice_dict(ds)]
        return out
