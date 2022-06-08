from gt4py.gtscript import PARALLEL, computation, interval

import pace.dsl.gt4py_utils as utils
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldI, FloatFieldIJ
from pace.util import TilePartitioner
from pace.util.grid import DriverGridData


def update_dwind_prep_stencil(
    u_dt: FloatField,
    v_dt: FloatField,
    vlon1: FloatFieldIJ,
    vlon2: FloatFieldIJ,
    vlon3: FloatFieldIJ,
    vlat1: FloatFieldIJ,
    vlat2: FloatFieldIJ,
    vlat3: FloatFieldIJ,
    ue_1: FloatField,
    ue_2: FloatField,
    ue_3: FloatField,
    ve_1: FloatField,
    ve_2: FloatField,
    ve_3: FloatField,
):
    with computation(PARALLEL), interval(...):
        v3_1 = u_dt * vlon1 + v_dt * vlat1
        v3_2 = u_dt * vlon2 + v_dt * vlat2
        v3_3 = u_dt * vlon3 + v_dt * vlat3
        ue_1 = v3_1[0, -1, 0] + v3_1
        ue_2 = v3_2[0, -1, 0] + v3_2
        ue_3 = v3_3[0, -1, 0] + v3_3
        ve_1 = v3_1[-1, 0, 0] + v3_1
        ve_2 = v3_2[-1, 0, 0] + v3_2
        ve_3 = v3_3[-1, 0, 0] + v3_3
    with computation(PARALLEL), interval(...):
        u_dt = 0.0
        v_dt = 0.0


def update_dwind_y_edge_south_stencil(
    ve_1: FloatField,
    ve_2: FloatField,
    ve_3: FloatField,
    vt_1: FloatField,
    vt_2: FloatField,
    vt_3: FloatField,
    edge_vect: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        vt_1 = edge_vect * ve_1[0, 1, 0] + (1.0 - edge_vect) * ve_1
        vt_2 = edge_vect * ve_2[0, 1, 0] + (1.0 - edge_vect) * ve_2
        vt_3 = edge_vect * ve_3[0, 1, 0] + (1.0 - edge_vect) * ve_3


def update_dwind_y_edge_north_stencil(
    ve_1: FloatField,
    ve_2: FloatField,
    ve_3: FloatField,
    vt_1: FloatField,
    vt_2: FloatField,
    vt_3: FloatField,
    edge_vect: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        vt_1 = edge_vect * ve_1[0, -1, 0] + (1.0 - edge_vect) * ve_1
        vt_2 = edge_vect * ve_2[0, -1, 0] + (1.0 - edge_vect) * ve_2
        vt_3 = edge_vect * ve_3[0, -1, 0] + (1.0 - edge_vect) * ve_3


def update_dwind_x_edge_west_stencil(
    ue_1: FloatField,
    ue_2: FloatField,
    ue_3: FloatField,
    ut_1: FloatField,
    ut_2: FloatField,
    ut_3: FloatField,
    edge_vect: FloatFieldI,
):
    with computation(PARALLEL), interval(...):
        ut_1 = edge_vect * ue_1[1, 0, 0] + (1.0 - edge_vect) * ue_1
        ut_2 = edge_vect * ue_2[1, 0, 0] + (1.0 - edge_vect) * ue_2
        ut_3 = edge_vect * ue_3[1, 0, 0] + (1.0 - edge_vect) * ue_3


def update_dwind_x_edge_east_stencil(
    ue_1: FloatField,
    ue_2: FloatField,
    ue_3: FloatField,
    ut_1: FloatField,
    ut_2: FloatField,
    ut_3: FloatField,
    edge_vect: FloatFieldI,
):
    with computation(PARALLEL), interval(...):
        ut_1 = edge_vect * ue_1[-1, 0, 0] + (1.0 - edge_vect) * ue_1
        ut_2 = edge_vect * ue_2[-1, 0, 0] + (1.0 - edge_vect) * ue_2
        ut_3 = edge_vect * ue_3[-1, 0, 0] + (1.0 - edge_vect) * ue_3


def copy3_stencil(
    in_field1: FloatField,
    in_field2: FloatField,
    in_field3: FloatField,
    out_field1: FloatField,
    out_field2: FloatField,
    out_field3: FloatField,
):
    with computation(PARALLEL), interval(...):
        out_field1 = in_field1
        out_field2 = in_field2
        out_field3 = in_field3


def update_uwind_stencil(
    u: FloatField,
    es1_1: FloatFieldIJ,
    es1_2: FloatFieldIJ,
    es1_3: FloatFieldIJ,
    ue_1: FloatField,
    ue_2: FloatField,
    ue_3: FloatField,
    dt5: float,
):
    with computation(PARALLEL), interval(...):
        # is: ie; js:je+1
        u = u + dt5 * (ue_1 * es1_1 + ue_2 * es1_2 + ue_3 * es1_3)


def update_vwind_stencil(
    v: FloatField,
    ew2_1: FloatFieldIJ,
    ew2_2: FloatFieldIJ,
    ew2_3: FloatFieldIJ,
    ve_1: FloatField,
    ve_2: FloatField,
    ve_3: FloatField,
    dt5: float,
):
    with computation(PARALLEL), interval(...):
        # is: ie+1; js:je
        v = v + dt5 * (ve_1 * ew2_1 + ve_2 * ew2_2 + ve_3 * ew2_3)


class AGrid2DGridPhysics:
    """
    Fortran name is update_dwinds_phys
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        partitioner: TilePartitioner,
        rank: int,
        namelist,
        grid_info: DriverGridData,
    ):
        grid_indexing = stencil_factory.grid_indexing
        self.namelist = namelist
        self._dt5 = 0.5 * self.namelist.dt_atmos
        npx = self.namelist.npx
        npy = self.namelist.npy
        self._im2 = int((npx - 1) / 2) + 2
        self._jm2 = int((npy - 1) / 2) + 2
        self._subtile_index = partitioner.tile.subtile_index(rank)
        layout = self.namelist.layout

        self._subtile_width_x = int((npx - 1) / layout[0])
        self._subtile_width_y = int((npy - 1) / layout[1])
        shape = grid_indexing.max_shape

        nic = grid_indexing.iec - grid_indexing.isc + 1
        njc = grid_indexing.jec - grid_indexing.jsc + 1
        npz = grid_indexing.domain[2]
        self.south_edge = grid_indexing.south_edge
        self.north_edge = grid_indexing.north_edge
        self.west_edge = grid_indexing.west_edge
        self.east_edge = grid_indexing.east_edge

        def make_storage():
            return utils.make_storage_from_shape(shape, backend=stencil_factory.backend)

        self._ue_1 = make_storage()
        self._ue_2 = make_storage()
        self._ue_3 = make_storage()
        self._ut_1 = make_storage()
        self._ut_2 = make_storage()
        self._ut_3 = make_storage()
        self._ve_1 = make_storage()
        self._ve_2 = make_storage()
        self._ve_3 = make_storage()
        self._vt_1 = make_storage()
        self._vt_2 = make_storage()
        self._vt_3 = make_storage()

        self._update_dwind_prep_stencil = stencil_factory.from_origin_domain(
            update_dwind_prep_stencil,
            origin=(grid_indexing.n_halo - 1, grid_indexing.n_halo - 1, 0),
            domain=(nic + 2, njc + 2, npz),
        )

        self.global_is, self.global_js = self.local_to_global_indices(
            grid_indexing.isc, grid_indexing.jsc
        )
        self.global_ie, self.global_je = self.local_to_global_indices(
            grid_indexing.iec, grid_indexing.jec
        )

        if self.west_edge:
            je_lower = self.global_to_local_y(min(self._jm2, self.global_je))
            origin_lower = (grid_indexing.n_halo, grid_indexing.n_halo, 0)
            self._domain_lower_west = (
                1,
                je_lower - grid_indexing.jsc + 1,
                npz,
            )
            if self.global_js <= self._jm2:
                if self._domain_lower_west[1] > 0:
                    self._update_dwind_y_edge_south_stencil1 = (
                        stencil_factory.from_origin_domain(
                            update_dwind_y_edge_south_stencil,
                            origin=origin_lower,
                            domain=self._domain_lower_west,
                        )
                    )
            if self.global_je > self._jm2:
                js_upper = self.global_to_local_y(max(self._jm2 + 1, self.global_js))
                origin_upper = (grid_indexing.n_halo, js_upper, 0)
                self._domain_upper_west = (
                    1,
                    grid_indexing.jec - js_upper + 1,
                    npz,
                )
                if self._domain_upper_west[1] > 0:
                    self._update_dwind_y_edge_north_stencil1 = (
                        stencil_factory.from_origin_domain(
                            update_dwind_y_edge_north_stencil,
                            origin=origin_upper,
                            domain=self._domain_upper_west,
                        )
                    )
                    self._copy3_stencil1 = stencil_factory.from_origin_domain(
                        copy3_stencil,
                        origin=origin_upper,
                        domain=self._domain_upper_west,
                    )
            if self.global_js <= self._jm2 and self._domain_lower_west[1] > 0:
                self._copy3_stencil2 = stencil_factory.from_origin_domain(
                    copy3_stencil, origin=origin_lower, domain=self._domain_lower_west
                )
        if self.east_edge:
            i_origin = shape[0] - grid_indexing.n_halo - 1
            je_lower = self.global_to_local_y(min(self._jm2, self.global_je))
            origin_lower = (i_origin, grid_indexing.n_halo, 0)
            self._domain_lower_east = (
                1,
                je_lower - grid_indexing.jsc + 1,
                npz,
            )
            if self.global_js <= self._jm2:
                if self._domain_lower_east[1] > 0:
                    self._update_dwind_y_edge_south_stencil2 = (
                        stencil_factory.from_origin_domain(
                            update_dwind_y_edge_south_stencil,
                            origin=origin_lower,
                            domain=self._domain_lower_east,
                        )
                    )

            if self.global_je > self._jm2:
                js_upper = self.global_to_local_y(max(self._jm2 + 1, self.global_js))
                origin_upper = (i_origin, js_upper, 0)
                self._domain_upper_east = (
                    1,
                    grid_indexing.jec - js_upper + 1,
                    npz,
                )
                if self._domain_upper_east[1] > 0:
                    self._update_dwind_y_edge_north_stencil2 = (
                        stencil_factory.from_origin_domain(
                            update_dwind_y_edge_north_stencil,
                            origin=origin_upper,
                            domain=self._domain_upper_east,
                        )
                    )
                    self._copy3_stencil3 = stencil_factory.from_origin_domain(
                        copy3_stencil,
                        origin=origin_upper,
                        domain=self._domain_upper_east,
                    )
            if self.global_js <= self._jm2 and self._domain_lower_east[1] > 0:
                self._copy3_stencil4 = stencil_factory.from_origin_domain(
                    copy3_stencil, origin=origin_lower, domain=self._domain_lower_east
                )
        if self.south_edge:
            ie_lower = self.global_to_local_x(min(self._im2, self.global_ie))
            origin_lower = (grid_indexing.n_halo, grid_indexing.n_halo, 0)
            self._domain_lower_south = (
                ie_lower - grid_indexing.isc + 1,
                1,
                npz,
            )
            if self.global_is <= self._im2:
                if self._domain_lower_south[0] > 0:
                    self._update_dwind_x_edge_west_stencil1 = (
                        stencil_factory.from_origin_domain(
                            update_dwind_x_edge_west_stencil,
                            origin=origin_lower,
                            domain=self._domain_lower_south,
                        )
                    )
            if self.global_ie > self._im2:
                is_upper = self.global_to_local_x(max(self._im2 + 1, self.global_is))
                origin_upper = (is_upper, grid_indexing.n_halo, 0)
                self._domain_upper_south = (
                    grid_indexing.iec - is_upper + 1,
                    1,
                    npz,
                )
                self._update_dwind_x_edge_east_stencil1 = (
                    stencil_factory.from_origin_domain(
                        update_dwind_x_edge_east_stencil,
                        origin=origin_upper,
                        domain=self._domain_upper_south,
                    )
                )
                self._copy3_stencil5 = stencil_factory.from_origin_domain(
                    copy3_stencil, origin=origin_upper, domain=self._domain_upper_south
                )
            if self.global_is <= self._im2 and self._domain_lower_south[0] > 0:
                self._copy3_stencil6 = stencil_factory.from_origin_domain(
                    copy3_stencil, origin=origin_lower, domain=self._domain_lower_south
                )
        if self.north_edge:
            j_origin = shape[1] - grid_indexing.n_halo - 1
            ie_lower = self.global_to_local_x(min(self._im2, self.global_ie))
            origin_lower = (grid_indexing.n_halo, j_origin, 0)
            self._domain_lower_north = (
                ie_lower - grid_indexing.isc + 1,
                1,
                npz,
            )
            if self.global_is < self._im2:
                if self._domain_lower_north[0] > 0:
                    self._update_dwind_x_edge_west_stencil2 = (
                        stencil_factory.from_origin_domain(
                            update_dwind_x_edge_west_stencil,
                            origin=origin_lower,
                            domain=self._domain_lower_north,
                        )
                    )
            if self.global_ie >= self._im2:
                is_upper = self.global_to_local_x(max(self._im2 + 1, self.global_is))
                origin_upper = (is_upper, j_origin, 0)
                self._domain_upper_north = (
                    grid_indexing.iec - is_upper + 1,
                    1,
                    npz,
                )
                if self._domain_upper_north[0] > 0:
                    self._update_dwind_x_edge_east_stencil2 = (
                        stencil_factory.from_origin_domain(
                            update_dwind_x_edge_east_stencil,
                            origin=origin_upper,
                            domain=self._domain_upper_north,
                        )
                    )
                    self._copy3_stencil7 = stencil_factory.from_origin_domain(
                        copy3_stencil,
                        origin=origin_upper,
                        domain=self._domain_upper_north,
                    )
            if self.global_is < self._im2 and self._domain_lower_north[0] > 0:
                self._copy3_stencil8 = stencil_factory.from_origin_domain(
                    copy3_stencil, origin=origin_lower, domain=self._domain_lower_north
                )
        self._update_uwind_stencil = stencil_factory.from_origin_domain(
            update_uwind_stencil,
            origin=(grid_indexing.n_halo, grid_indexing.n_halo, 0),
            domain=(nic, njc + 1, npz),
        )
        self._update_vwind_stencil = stencil_factory.from_origin_domain(
            update_vwind_stencil,
            origin=(grid_indexing.n_halo, grid_indexing.n_halo, 0),
            domain=(nic + 1, njc, npz),
        )
        # [TODO] The following is waiting on grid code vlat and vlon
        self._vlon1 = grid_info.vlon1
        self._vlon2 = grid_info.vlon2
        self._vlon3 = grid_info.vlon3
        self._vlat1 = grid_info.vlat1
        self._vlat2 = grid_info.vlat2
        self._vlat3 = grid_info.vlat3
        self._edge_vect_w = grid_info.edge_vect_w
        self._edge_vect_e = grid_info.edge_vect_e
        self._edge_vect_s = grid_info.edge_vect_s
        self._edge_vect_n = grid_info.edge_vect_n
        self._es1_1 = grid_info.es1_1
        self._es1_2 = grid_info.es1_2
        self._es1_3 = grid_info.es1_3
        self._ew2_1 = grid_info.ew2_1
        self._ew2_2 = grid_info.ew2_2
        self._ew2_3 = grid_info.ew2_3

    def global_to_local_1d(self, global_value, subtile_index, subtile_length):
        return global_value - subtile_index * subtile_length

    def global_to_local_y(self, j_global):
        return self.global_to_local_1d(
            j_global, self._subtile_index[0], self._subtile_width_y
        )

    def global_to_local_x(self, i_global):
        return self.global_to_local_1d(
            i_global, self._subtile_index[1], self._subtile_width_x
        )

    def local_to_global_indices(self, i_local, j_local):
        i_global = self.local_to_global_1d(
            i_local, self._subtile_index[1], self._subtile_width_x
        )
        j_global = self.local_to_global_1d(
            j_local, self._subtile_index[0], self._subtile_width_y
        )
        return i_global, j_global

    def local_to_global_1d(self, local_value, subtile_index, subtile_length):
        return local_value + subtile_index * subtile_length

    def __call__(
        self,
        u: FloatField,
        v: FloatField,
        u_dt: FloatField,
        v_dt: FloatField,
    ):
        """
        Transforms the wind tendencies from A grid to D grid for the final update
        """

        self._update_dwind_prep_stencil(
            u_dt,
            v_dt,
            self._vlon1,
            self._vlon2,
            self._vlon3,
            self._vlat1,
            self._vlat2,
            self._vlat3,
            self._ue_1,
            self._ue_2,
            self._ue_3,
            self._ve_1,
            self._ve_2,
            self._ve_3,
        )
        if self.west_edge:
            if self.global_js <= self._jm2:
                if self._domain_lower_west[1] > 0:
                    self._update_dwind_y_edge_south_stencil1(
                        self._ve_1,
                        self._ve_2,
                        self._ve_3,
                        self._vt_1,
                        self._vt_2,
                        self._vt_3,
                        self._edge_vect_w,
                    )
            if self.global_je > self._jm2:
                if self._domain_upper_west[1] > 0:
                    self._update_dwind_y_edge_north_stencil1(
                        self._ve_1,
                        self._ve_2,
                        self._ve_3,
                        self._vt_1,
                        self._vt_2,
                        self._vt_3,
                        self._edge_vect_w,
                    )
                    self._copy3_stencil1(
                        self._vt_1,
                        self._vt_2,
                        self._vt_3,
                        self._ve_1,
                        self._ve_2,
                        self._ve_3,
                    )
            if self.global_js <= self._jm2 and self._domain_lower_west[1] > 0:
                self._copy3_stencil2(
                    self._vt_1,
                    self._vt_2,
                    self._vt_3,
                    self._ve_1,
                    self._ve_2,
                    self._ve_3,
                )
        if self.east_edge:
            if self.global_js <= self._jm2:
                if self._domain_lower_east[1] > 0:
                    self._update_dwind_y_edge_south_stencil2(
                        self._ve_1,
                        self._ve_2,
                        self._ve_3,
                        self._vt_1,
                        self._vt_2,
                        self._vt_3,
                        self._edge_vect_e,
                    )
            if self.global_je > self._jm2:
                if self._domain_upper_east[1] > 0:
                    self._update_dwind_y_edge_north_stencil2(
                        self._ve_1,
                        self._ve_2,
                        self._ve_3,
                        self._vt_1,
                        self._vt_2,
                        self._vt_3,
                        self._edge_vect_e,
                    )
                    self._copy3_stencil3(
                        self._vt_1,
                        self._vt_2,
                        self._vt_3,
                        self._ve_1,
                        self._ve_2,
                        self._ve_3,
                    )
            if self.global_js <= self._jm2 and self._domain_lower_east[1] > 0:
                self._copy3_stencil4(
                    self._vt_1,
                    self._vt_2,
                    self._vt_3,
                    self._ve_1,
                    self._ve_2,
                    self._ve_3,
                )
        if self.south_edge:
            if self.global_is <= self._im2:
                if self._domain_lower_south[0] > 0:
                    self._update_dwind_x_edge_west_stencil1(
                        self._ue_1,
                        self._ue_2,
                        self._ue_3,
                        self._ut_1,
                        self._ut_2,
                        self._ut_3,
                        self._edge_vect_s,
                    )
            if self.global_ie > self._im2:
                if self._domain_upper_south:
                    self._update_dwind_x_edge_east_stencil1(
                        self._ue_1,
                        self._ue_2,
                        self._ue_3,
                        self._ut_1,
                        self._ut_2,
                        self._ut_3,
                        self._edge_vect_s,
                    )
                    self._copy3_stencil5(
                        self._ut_1,
                        self._ut_2,
                        self._ut_3,
                        self._ue_1,
                        self._ue_2,
                        self._ue_3,
                    )
            if self.global_is <= self._im2 and self._domain_lower_south[0] > 0:
                self._copy3_stencil6(
                    self._ut_1,
                    self._ut_2,
                    self._ut_3,
                    self._ue_1,
                    self._ue_2,
                    self._ue_3,
                )
        if self.north_edge:
            if self.global_is < self._im2:
                if self._domain_lower_north[0] > 0:
                    self._update_dwind_x_edge_west_stencil2(
                        self._ue_1,
                        self._ue_2,
                        self._ue_3,
                        self._ut_1,
                        self._ut_2,
                        self._ut_3,
                        self._edge_vect_n,
                    )
            if self.global_ie >= self._im2:
                if self._domain_upper_north[0] > 0:
                    self._update_dwind_x_edge_east_stencil2(
                        self._ue_1,
                        self._ue_2,
                        self._ue_3,
                        self._ut_1,
                        self._ut_2,
                        self._ut_3,
                        self._edge_vect_n,
                    )
                    self._copy3_stencil7(
                        self._ut_1,
                        self._ut_2,
                        self._ut_3,
                        self._ue_1,
                        self._ue_2,
                        self._ue_3,
                    )
            if self.global_is < self._im2 and self._domain_lower_north[0] > 0:
                self._copy3_stencil8(
                    self._ut_1,
                    self._ut_2,
                    self._ut_3,
                    self._ue_1,
                    self._ue_2,
                    self._ue_3,
                )
        self._update_uwind_stencil(
            u,
            self._es1_1,
            self._es1_2,
            self._es1_3,
            self._ue_1,
            self._ue_2,
            self._ue_3,
            self._dt5,
        )
        self._update_vwind_stencil(
            v,
            self._ew2_1,
            self._ew2_2,
            self._ew2_3,
            self._ve_1,
            self._ve_2,
            self._ve_3,
            self._dt5,
        )
