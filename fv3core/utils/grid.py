from typing import Optional, Tuple

import fv3gfs.util as fv3util
import numpy as np

import fv3core.utils.gt4py_utils as utils


class Grid:
    # indices = ["is_", "ie", "isd", "ied", "js", "je", "jsd", "jed"]
    index_pairs = [("is_", "js"), ("ie", "je"), ("isd", "jsd"), ("ied", "jed")]
    shape_params = ["npz", "npx", "npy"]
    # npx -- number of grid corners on one tile of the domain
    # grid.ie == npx - 1identified east edge in fortran
    # But we need to add the halo - 1 to change this check to 0 based python arrays
    # grid.ie == npx + halo - 2

    def __init__(self, indices, shape_params, rank, layout, data_fields={}):
        self.rank = rank
        self.partitioner = fv3util.TilePartitioner(layout)
        self.subtile_index = self.partitioner.subtile_index(self.rank)
        self.layout = layout
        for s in self.shape_params:
            setattr(self, s, int(shape_params[s]))
        self.subtile_width_x = int((self.npx - 1) / self.layout[0])
        self.subtile_width_y = int((self.npy - 1) / self.layout[1])
        for ivar, jvar in self.index_pairs:
            local_i, local_j = self.global_to_local_indices(
                int(indices[ivar]), int(indices[jvar])
            )
            setattr(self, ivar, local_i)
            setattr(self, jvar, local_j)
        self.nid = int(self.ied - self.isd + 1)
        self.njd = int(self.jed - self.jsd + 1)
        self.nic = int(self.ie - self.is_ + 1)
        self.njc = int(self.je - self.js + 1)
        self.halo = utils.halo
        # TODO: do we want to set face indices this way?
        self.isf = 0
        self.ief = self.npx - 1
        self.jsf = 0
        self.jef = self.npy - 1
        self.global_is, self.global_js = self.local_to_global_indices(self.is_, self.js)
        self.global_ie, self.global_je = self.local_to_global_indices(self.ie, self.je)
        self.global_isd, self.global_jsd = self.local_to_global_indices(
            self.isd, self.jsd
        )
        self.global_ied, self.global_jed = self.local_to_global_indices(
            self.ied, self.jed
        )
        self.west_edge = self.global_is == self.halo
        self.east_edge = self.global_ie == self.npx + self.halo - 2
        self.south_edge = self.global_js == self.halo
        self.north_edge = self.global_je == self.npy + self.halo - 2

        self.j_offset = self.js - self.jsd - 1
        self.i_offset = self.is_ - self.isd - 1
        self.sw_corner = self.west_edge and self.south_edge
        self.se_corner = self.east_edge and self.south_edge
        self.nw_corner = self.west_edge and self.north_edge
        self.ne_corner = self.east_edge and self.north_edge
        self.data_fields = {}
        self.add_data(data_fields)
        self._sizer = None
        self._quantity_factory = None

    @property
    def sizer(self):
        if self._sizer is None:
            # in the future this should use from_namelist, when we have a non-flattened
            # namelist
            self._sizer = fv3util.SubtileGridSizer.from_tile_params(
                nx_tile=self.npx - 1,
                ny_tile=self.npy - 1,
                nz=self.npz,
                n_halo=self.halo,
                extra_dim_lengths={},
                layout=self.layout,
            )
        return self._sizer

    @property
    def quantity_factory(self):
        if self._quantity_factory is None:
            self._quantity_factory = fv3util.QuantityFactory.from_backend(
                self.sizer, backend=utils.backend
            )
        return self._quantity_factory

    def splitters(self, *, origin=None):
        """Return the splitters relative to origin.

        Args:
            origin: The compute origin

        """
        if origin is None:
            origin = self.compute_origin()
        return {
            "i_start": self.is_ - self.global_is + (self.is_ - origin[0]),
            "i_end": self.npx + self.halo - 2 - self.global_is + (self.is_ - origin[0]),
            "j_start": self.js - self.global_js + (self.js - origin[1]),
            "j_end": self.npy + self.halo - 2 - self.global_js + (self.js - origin[1]),
        }

    def make_quantity(
        self,
        array,
        dims=[fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
        units="Unknown",
        origin=None,
        extent=None,
    ):
        if origin is None:
            origin = self.compute_origin()
        if extent is None:
            extent = self.domain_shape_compute()
        return fv3util.Quantity(
            array, dims=dims, units=units, origin=origin, extent=extent
        )

    def quantity_dict_update(
        self,
        data_dict,
        varname,
        dims=[fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
        units="Unknown",
    ):
        data_dict[varname + "_quantity"] = self.quantity_wrap(
            data_dict[varname], dims=dims, units=units
        )

    def quantity_wrap(
        self, data, dims=[fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], units="Unknown"
    ):
        origin = self.sizer.get_origin(dims)
        extent = self.sizer.get_extent(dims)
        return fv3util.Quantity(
            data, dims=dims, units=units, origin=origin, extent=extent
        )

    def global_to_local_1d(self, global_value, subtile_index, subtile_length):
        return global_value - subtile_index * subtile_length

    def global_to_local_x(self, i_global):
        return self.global_to_local_1d(
            i_global, self.subtile_index[1], self.subtile_width_x
        )

    def global_to_local_y(self, j_global):
        return self.global_to_local_1d(
            j_global, self.subtile_index[0], self.subtile_width_y
        )

    def global_to_local_indices(self, i_global, j_global):
        i_local = self.global_to_local_x(i_global)
        j_local = self.global_to_local_y(j_global)
        return i_local, j_local

    def local_to_global_1d(self, local_value, subtile_index, subtile_length):
        return local_value + subtile_index * subtile_length

    def local_to_global_indices(self, i_local, j_local):
        i_global = self.local_to_global_1d(
            i_local, self.subtile_index[1], self.subtile_width_x
        )
        j_global = self.local_to_global_1d(
            j_local, self.subtile_index[0], self.subtile_width_y
        )
        return i_global, j_global

    def add_data(self, data_dict):
        self.data_fields.update(data_dict)
        for k, v in self.data_fields.items():
            setattr(self, k, v)

    def irange_compute(self):
        return range(self.is_, self.ie + 1)

    def irange_compute_x(self):
        return range(self.is_, self.ie + 2)

    def jrange_compute(self):
        return range(self.js, self.je + 1)

    def jrange_compute_y(self):
        return range(self.js, self.je + 2)

    def irange_domain(self):
        return range(self.isd, self.ied + 1)

    def jrange_domain(self):
        return range(self.jsd, self.jed + 1)

    def krange(self):
        return range(0, self.npz)

    def compute_interface(self):
        return self.slice_dict(self.compute_dict())

    def x3d_interface(self):
        return self.slice_dict(self.x3d_compute_dict())

    def y3d_interface(self):
        return self.slice_dict(self.y3d_compute_dict())

    def x3d_domain_interface(self):
        return self.slice_dict(self.x3d_domain_dict())

    def y3d_domain_interface(self):
        return self.slice_dict(self.y3d_domain_dict())

    def add_one(self, num):
        if num is None:
            return None
        return num + 1

    def slice_dict(self, d):
        return (
            slice(d["istart"], self.add_one(d["iend"])),
            slice(d["jstart"], self.add_one(d["jend"])),
            slice(d["kstart"], self.add_one(d["kend"])),
        )

    def default_domain_dict(self):
        return {
            "istart": self.isd,
            "iend": self.ied,
            "jstart": self.jsd,
            "jend": self.jed,
            "kstart": 0,
            "kend": self.npz - 1,
        }

    def default_dict_buffer_2d(self):
        mydict = self.default_domain_dict()
        mydict["iend"] += 1
        mydict["jend"] += 1
        return mydict

    def compute_dict(self):
        return {
            "istart": self.is_,
            "iend": self.ie,
            "jstart": self.js,
            "jend": self.je,
            "kstart": 0,
            "kend": self.npz - 1,
        }

    def compute_dict_buffer_2d(self):
        mydict = self.compute_dict()
        mydict["iend"] += 1
        mydict["jend"] += 1
        return mydict

    def default_buffer_k_dict(self):
        mydict = self.default_domain_dict()
        mydict["kend"] = self.npz
        return mydict

    def compute_buffer_k_dict(self):
        mydict = self.compute_dict()
        mydict["kend"] = self.npz
        return mydict

    def x3d_domain_dict(self):
        horizontal_dict = {
            "istart": self.isd,
            "iend": self.ied + 1,
            "jstart": self.jsd,
            "jend": self.jed,
        }
        return {**self.default_domain_dict(), **horizontal_dict}

    def y3d_domain_dict(self):
        horizontal_dict = {
            "istart": self.isd,
            "iend": self.ied,
            "jstart": self.jsd,
            "jend": self.jed + 1,
        }
        return {**self.default_domain_dict(), **horizontal_dict}

    def x3d_compute_dict(self):
        horizontal_dict = {
            "istart": self.is_,
            "iend": self.ie + 1,
            "jstart": self.js,
            "jend": self.je,
        }
        return {**self.default_domain_dict(), **horizontal_dict}

    def y3d_compute_dict(self):
        horizontal_dict = {
            "istart": self.is_,
            "iend": self.ie,
            "jstart": self.js,
            "jend": self.je + 1,
        }
        return {**self.default_domain_dict(), **horizontal_dict}

    def x3d_compute_domain_y_dict(self):
        horizontal_dict = {
            "istart": self.is_,
            "iend": self.ie + 1,
            "jstart": self.jsd,
            "jend": self.jed,
        }
        return {**self.default_domain_dict(), **horizontal_dict}

    def y3d_compute_domain_x_dict(self):
        horizontal_dict = {
            "istart": self.isd,
            "iend": self.ied,
            "jstart": self.js,
            "jend": self.je + 1,
        }
        return {**self.default_domain_dict(), **horizontal_dict}

    def domain_shape_standard(self):
        return (self.nid, self.njd, self.npz)

    def domain_shape_buffer_k(self):
        return (self.nid, self.njd, self.npz + 1)

    def domain_shape_compute(self):
        return (self.nic, self.njc, self.npz)

    def domain_shape_compute_buffer_2d(self, add: Tuple[int, int, int] = (1, 1, 0)):
        return (self.nic + add[0], self.njc + add[1], self.npz + add[2])

    def domain_shape_compute_buffer_k(self):
        return (self.nic, self.njc, self.npz + 1)

    def domain_shape_compute_x(self):
        return (self.nic + 1, self.njc, self.npz)

    def domain_shape_compute_y(self):
        return (self.nic, self.njc + 1, self.npz)

    def domain_x_compute_y(self):
        return (self.nid, self.njc, self.npz)

    def domain_x_compute_ybuffer(self):
        return (self.nid, self.njc + 1, self.npz)

    def domain_y_compute_x(self):
        return (self.nic, self.njd, self.npz)

    def domain_y_compute_xbuffer(self):
        return (self.nic + 1, self.njd, self.npz)

    def domain_shape_buffer_1cell(self):
        return (int(self.nid + 1), int(self.njd + 1), int(self.npz + 1))

    def domain2d_ik_buffer_1cell(self):
        return (int(self.nid + 1), 1, int(self.npz + 1))

    def domain_shape_y(self):
        return (int(self.nid), int(self.njd + 1), int(self.npz))

    def domain_shape_x(self):
        return (int(self.nid + 1), int(self.njd), int(self.npz))

    def corner_domain(self):
        return (1, 1, self.npz)

    def domain_shape_buffer_2d(self):
        return (int(self.nid + 1), int(self.njd + 1), int(self.npz))

    def copy_right_edge(self, var, i_index, j_index):
        return np.copy(var[i_index:, :, :]), np.copy(var[:, j_index:, :])

    def insert_left_edge(self, var, edge_data_i, i_index, edge_data_j, j_index):
        var[:i_index, :, :] = edge_data_i
        var[:, :j_index, :] = edge_data_j

    def insert_right_edge(self, var, edge_data_i, i_index, edge_data_j, j_index):
        var[i_index:, :, :] = edge_data_i
        var[:, j_index:, :] = edge_data_j

    def uvar_edge_halo(self, var):
        return self.copy_right_edge(var, self.ie + 2, self.je + 1)

    def vvar_edge_halo(self, var):
        return self.copy_right_edge(var, self.ie + 1, self.je + 2)

    def edge_offset_halos(self, uvar, vvar):
        u_edge_i, u_edge_j = self.uvar_edge_halo(uvar)
        v_edge_i, v_edge_j = self.vvar_edge_halo(vvar)
        return u_edge_i, u_edge_j, v_edge_i, v_edge_j

    def insert_edge(self, var, edge_data, index):
        var[index] = edge_data

    def append_edges(self, uvar, u_edge_i, u_edge_j, vvar, v_edge_i, v_edge_j):
        self.insert_right_edge(uvar, u_edge_i, self.ie + 2, u_edge_j, self.je + 1)
        self.insert_right_edge(vvar, v_edge_i, self.ie + 1, v_edge_j, self.je + 2)

    def overwrite_edges(self, var, edgevar, left_i_index, left_j_index):
        self.insert_left_edge(
            var,
            edgevar[:left_i_index, :, :],
            left_i_index,
            edgevar[:, :left_j_index, :],
            left_j_index,
        )
        right_i_index = self.ie + left_i_index
        right_j_index = self.ie + left_j_index
        self.insert_right_edge(
            var,
            edgevar[right_i_index:, :, :],
            right_i_index,
            edgevar[:, right_j_index:, :],
            right_j_index,
        )

    def compute_origin(self, add: Tuple[int, int, int] = (0, 0, 0)):
        return (self.is_ + add[0], self.js + add[1], add[2])

    def default_origin(self):
        return (self.isd, self.jsd, 0)

    def compute_x_origin(self):
        return (self.is_, self.jsd, 0)

    def compute_y_origin(self):
        return (self.isd, self.js, 0)

    # TODO, expand to more cases
    def horizontal_starts_from_shape(self, shape):
        if shape[0:2] in [
            self.domain_shape_compute()[0:2],
            self.domain_shape_compute_x()[0:2],
            self.domain_shape_compute_y()[0:2],
            self.domain_shape_compute_buffer_2d()[0:2],
        ]:
            return self.is_, self.js
        elif shape[0:2] == (self.nic + 2, self.njc + 2):
            return self.is_ - 1, self.js - 1
        else:
            return 0, 0

    def slice_data_k(self, ki):
        utils.k_slice_inplace(self.data_fields, ki)
        # update instance vars
        for k, v in self.data_fields.items():
            setattr(self, k, self.data_fields[k])
