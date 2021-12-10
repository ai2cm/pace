from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.d_sw as d_sw
from fv3core.testing import TranslateFortranData2Py
from fv3core.utils.grid import axis_offsets
from pace.dsl.typing import FloatField, FloatFieldIJ


class TranslateD_SW(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.max_error = 3.2e-10
        column_namelist = d_sw.get_column_namelist(
            spec.namelist, grid.npz, backend=self.grid.stencil_factory.backend
        )
        self.compute_func = d_sw.DGridShallowWaterLagrangianDynamics(
            spec.grid.stencil_factory,
            spec.grid.grid_data,
            spec.grid.damping_coefficients,
            column_namelist,
            nested=spec.grid.nested,
            stretched_grid=spec.grid.stretched_grid,
            config=spec.namelist.dynamical_core.acoustic_dynamics.d_grid_shallow_water,
        )
        self.in_vars["data_vars"] = {
            "uc": grid.x3d_domain_dict(),
            "vc": grid.y3d_domain_dict(),
            "w": {},
            "delpc": {},
            "delp": {},
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "xfx": grid.x3d_compute_domain_y_dict(),
            "crx": grid.x3d_compute_domain_y_dict(),
            "yfx": grid.y3d_compute_domain_x_dict(),
            "cry": grid.y3d_compute_domain_x_dict(),
            "mfx": grid.x3d_compute_dict(),
            "mfy": grid.y3d_compute_dict(),
            "cx": grid.x3d_compute_domain_y_dict(),
            "cy": grid.y3d_compute_domain_x_dict(),
            "heat_source": {},
            "diss_est": {},
            "q_con": {},
            "pt": {},
            "ptc": {},
            "ua": {},
            "va": {},
            "zh": {},
            "divgd": grid.default_dict_buffer_2d(),
        }
        for name, info in self.in_vars["data_vars"].items():
            info["serialname"] = name + "d"
        self.in_vars["parameters"] = ["dt"]
        self.out_vars = self.in_vars["data_vars"].copy()
        del self.out_vars["zh"]


def ubke(
    uc: FloatField,
    vc: FloatField,
    cosa: FloatFieldIJ,
    rsina: FloatFieldIJ,
    ut: FloatField,
    ub: FloatField,
    dt4: float,
    dt5: float,
):
    with computation(PARALLEL), interval(...):
        dt = 2.0 * dt5
        ub, _ = d_sw.interpolate_uc_vc_to_cell_corners(uc, vc, cosa, rsina, ut, ut)
        ub = ub * dt


class TranslateUbKE(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "ut": {},
            "ub": grid.compute_dict_buffer_2d(),
        }
        self.in_vars["parameters"] = ["dt5", "dt4"]
        self.out_vars = {"ub": grid.compute_dict_buffer_2d()}
        origin = self.grid.compute_origin()
        domain = self.grid.domain_shape_compute(add=(1, 1, 0))
        ax_offsets = axis_offsets(self.grid, origin, domain)
        self.compute_func = self.grid.stencil_factory.from_origin_domain(
            ubke, externals=ax_offsets, origin=origin, domain=domain
        )

    def compute_from_storage(self, inputs):
        inputs["cosa"] = self.grid.cosa
        inputs["rsina"] = self.grid.rsina
        self.compute_func(**inputs)
        return inputs


def vbke(
    vc: FloatField,
    uc: FloatField,
    cosa: FloatFieldIJ,
    rsina: FloatFieldIJ,
    vt: FloatField,
    vb: FloatField,
    dt4: float,
    dt5: float,
):
    with computation(PARALLEL), interval(...):
        dt = 2.0 * dt5
        _, vb = d_sw.interpolate_uc_vc_to_cell_corners(uc, vc, cosa, rsina, vt, vt)
        vb = vb * dt


class TranslateVbKE(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "vc": {},
            "uc": {},
            "vt": {},
            "vb": grid.compute_dict_buffer_2d(),
        }
        self.in_vars["parameters"] = ["dt5", "dt4"]
        self.out_vars = {"vb": grid.compute_dict_buffer_2d()}
        origin = self.grid.compute_origin()
        domain = self.grid.domain_shape_compute(add=(1, 1, 0))
        ax_offsets = axis_offsets(self.grid, origin, domain)
        self.compute_func = self.grid.stencil_factory.from_origin_domain(
            vbke, externals=ax_offsets, origin=origin, domain=domain
        )

    def compute_from_storage(self, inputs):
        inputs["cosa"] = self.grid.cosa
        inputs["rsina"] = self.grid.rsina
        self.compute_func(**inputs)
        return inputs


class TranslateFluxCapacitor(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "cx": grid.x3d_compute_domain_y_dict(),
            "cy": grid.y3d_compute_domain_x_dict(),
            "xflux": grid.x3d_compute_dict(),
            "yflux": grid.y3d_compute_dict(),
            "crx_adv": grid.x3d_compute_domain_y_dict(),
            "cry_adv": grid.y3d_compute_domain_x_dict(),
            "fx": grid.x3d_compute_dict(),
            "fy": grid.y3d_compute_dict(),
        }
        self.out_vars = {}
        for outvar in ["cx", "cy", "xflux", "yflux"]:
            self.out_vars[outvar] = self.in_vars["data_vars"][outvar]
        self.compute_func = grid.stencil_factory.from_origin_domain(
            d_sw.flux_capacitor,
            origin=grid.full_origin(),
            domain=grid.domain_shape_full(),
        )


class TranslateHeatDiss(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "fx2": {},
            "fy2": {},
            "w": {},
            "dw": {},
            "heat_source": {},
            "diss_est": {},
        }
        self.out_vars = {
            "heat_source": grid.compute_dict(),
            "diss_est": grid.compute_dict(),
            "dw": grid.compute_dict(),
        }

    def compute_from_storage(self, inputs):
        column_namelist = d_sw.get_column_namelist(
            spec.namelist, self.grid.npz, backend=self.grid.stencil_factory.backend
        )
        # TODO add these to the serialized data or remove the test
        inputs["damp_w"] = column_namelist["damp_w"]
        inputs["ke_bg"] = column_namelist["ke_bg"]
        inputs["dt"] = (
            spec.namelist.dt_atmos / spec.namelist.k_split / spec.namelist.n_split
        )
        inputs["rarea"] = self.grid.rarea
        heat_diss_stencil = self.grid.stencil_factory.from_origin_domain(
            d_sw.heat_diss,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(),
        )
        heat_diss_stencil(**inputs)
        return inputs


class TranslateWdivergence(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "q": {"serialname": "w"},
            "delp": {},
            "gx": {},
            "gy": {},
        }
        self.out_vars = {"q": {"serialname": "w"}}
        self.compute_func = self.grid.stencil_factory.from_origin_domain(
            d_sw.flux_adjust,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(),
        )

    def compute_from_storage(self, inputs):
        inputs["rarea"] = self.grid.rarea
        self.compute_func(**inputs)
        return inputs
