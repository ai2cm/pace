import dataclasses

import numpy as np

import fv3core._config as spec
import pace.dsl.gt4py_utils as utils
import pace.util
from fv3core.testing import ParallelTranslate2Py
from fv3gfs.physics.stencils.fv_update_phys import ApplyPhysics2Dycore
from pace.dsl.typing import FloatField, FloatFieldIJ


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


class TranslateFVUpdatePhys(ParallelTranslate2Py):
    def __init__(self, grids):
        super().__init__(grids)
        grid = grids[0]
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
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz,
                "kaxis": 1,
            },
            "delp": {},
            "pt": {},
            "ps": {},
            "pe": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
                "kaxis": 1,
            },
            "pk": grid.compute_buffer_k_dict(),
            "edge_vect_e": {"dwind": True},
            "edge_vect_n": {"dwind": True, "axis": 0},
            "edge_vect_s": {"dwind": True, "axis": 0},
            "edge_vect_w": {"dwind": True},
            "vlat": {"dwind": True},
            "vlon": {"dwind": True},
            "es": {"dwind": True},
            "ew": {"dwind": True},
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

    def add_composite_vvar_storage(self, d, var, data3d, max_shape, start_indices):
        """This function is needed to transform vlat, vlon, es, and ew
        This can be removed when the information is in Grid
        """
        shape = data3d.shape
        start1, start2 = start_indices.get(var, (0, 0))
        size1, size2 = data3d.shape[0:2]
        for s in range(data3d.shape[2]):
            buffer = np.zeros(max_shape[0:2])
            buffer[start1 : start1 + size1, start2 : start2 + size2] = np.squeeze(
                data3d[:, :, s]
            )
            d[var + str(s + 1)] = utils.make_storage_data(
                data=buffer,
                shape=max_shape[0:2],
                origin=(start1, start2),
                backend=self.grid.stencil_factory.backend,
            )
        d[var] = utils.make_storage_from_shape(
            shape=max_shape[0:2],
            origin=(start1, start2),
            init=True,
            backend=self.grid.stencil_factory.backend,
        )  # write the original name to avoid missing var

    def add_composite_evar_storage(self, d, var, data4d, max_shape, start_indices):
        shape = data4d.shape
        start1, start2 = start_indices.get(var, (0, 0))
        size1, size2 = data4d.shape[1:3]
        for s in range(data4d.shape[0]):
            for t in range(data4d.shape[3]):
                buffer = np.zeros(max_shape[0:2])
                buffer[start1 : start1 + size1, start2 : start2 + size2] = np.squeeze(
                    data4d[s, :, :, t]
                )
                d[var + str(s + 1) + "_" + str(t + 1)] = utils.make_storage_data(
                    data=buffer,
                    origin=(start1, start2),
                    shape=max_shape[0:2],
                    backend=self.grid.stencil_factory.backend,
                )
        d[var] = utils.make_storage_from_shape(
            shape=max_shape[0:2],
            origin=(start1, start2),
            init=True,
            backend=self.grid.stencil_factory.backend,
        )  # write the original name to avoid missing var

    def edge_vector_storage(self, d, var, axis):
        max_shape = self.grid.domain_shape_full(add=(1, 1, 1))
        default_origin = (0, 0, 0)
        if axis == 1:
            default_origin = (0, 0)
            d[var] = d[var][np.newaxis, ...]
            d[var] = np.repeat(d[var], max_shape[0], axis=0)
        if axis == 0:
            default_origin = (0,)
        d[var] = utils.make_storage_data(
            data=d[var],
            origin=default_origin,
            shape=d[var].shape,
            backend=self.grid.stencil_factory.backend,
        )

    def read_dwind_serialized_data(self, serializer, savepoint, varname):
        max_shape = self.grid.domain_shape_full(add=(1, 1, 1))
        start_indices = {
            "vlon": (self.grid.isd + 1, self.grid.jsd + 1),
            "vlat": (self.grid.isd + 1, self.grid.jsd + 1),
        }
        axes = {"edge_vect_s": 0, "edge_vect_n": 0, "edge_vect_w": 1, "edge_vect_e": 1}
        input_data = {}
        data = serializer.read(varname, savepoint)
        if varname in ["vlat", "vlon"]:
            self.add_composite_vvar_storage(
                input_data, varname, data, max_shape, start_indices
            )
            return input_data
        if varname in ["es", "ew"]:
            self.add_composite_evar_storage(
                input_data, varname, data, max_shape, start_indices
            )
            return input_data
        # convert single element numpy arrays to scalars
        elif data.size == 1:
            data = data.item()
            input_data[varname] = data
            return input_data
        elif len(data.shape) < 2:
            start1 = start_indices.get(varname, 0)
            size1 = data.shape[0]
            axis = axes.get(varname, 2)
            input_data[varname] = np.zeros(max_shape[axis])
            input_data[varname][start1 : start1 + size1] = data
            if "edge_vect" in varname:
                self.edge_vector_storage(input_data, varname, axis)
                return input_data
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
            origin=self.grid.full_origin(),
            shape=input_data[varname].shape,
            backend=self.grid.stencil_factory.backend,
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
        del inputs["vlat"]
        del inputs["vlon"]
        del inputs["es"]
        del inputs["ew"]
        del inputs["es1_2"]
        del inputs["es2_2"]
        del inputs["es3_2"]
        del inputs["ew1_1"]
        del inputs["ew2_1"]
        del inputs["ew3_1"]
        extra_grid_info = {}
        for key in [
            "edge_vect_e",
            "edge_vect_w",
            "edge_vect_s",
            "edge_vect_n",
            "vlat1",
            "vlat2",
            "vlat3",
            "vlon1",
            "vlon2",
            "vlon3",
            "es1_1",
            "es2_1",
            "es3_1",
            "ew1_2",
            "ew2_2",
            "ew3_2",
        ]:
            extra_grid_info[key] = inputs.pop(key)

        tendencies = {}
        for key in ["u_dt", "v_dt", "t_dt"]:
            tendencies[key] = inputs.pop(key)
        partitioner = pace.util.CubedSpherePartitioner(
            pace.util.TilePartitioner(spec.namelist.layout)
        )
        self._base.compute_func = ApplyPhysics2Dycore(
            self.grid.stencil_factory,
            self.grid.grid_data,
            spec.namelist,
            communicator,
            partitioner,
            self.grid.rank,
            extra_grid_info,
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
        state.u_quantity = u_quantity
        state.u = u_quantity.storage
        state.v_quantity = v_quantity
        state.v = v_quantity.storage
        self._base.compute_func(
            state,
            tendencies["u_dt"],
            tendencies["v_dt"],
            tendencies["t_dt"],
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
        state.u.synchronize()
        state.v.synchronize()
        state.ua.synchronize()
        state.va.synchronize()
        out["u"] = np.asarray(state.u)[self.grid.y3d_domain_interface()]
        out["v"] = np.asarray(state.v)[self.grid.x3d_domain_interface()]
        out["ua"] = np.asarray(state.ua)[self.grid.slice_dict(ds)]
        out["va"] = np.asarray(state.va)[self.grid.slice_dict(ds)]
        return out
