import pace.dsl
import pace.util
from fv3core.stencils.c_sw import CGridShallowWaterDynamics
from pace.stencils.testing import TranslateDycoreFortranData2Py


def get_c_sw_instance(
    grid, namelist: pace.util.Namelist, stencil_factory: pace.dsl.StencilFactory
):
    return CGridShallowWaterDynamics(
        stencil_factory,
        grid.grid_data,
        grid.nested,
        namelist.grid_type,
        namelist.nord,
    )


def compute_vorticitytransport_cgrid(
    c_sw: CGridShallowWaterDynamics,
    uc,
    vc,
    vort_c,
    ke_c,
    v,
    u,
    dt2: float,
):
    """Update the C-Grid x and y velocity fields.

    Args:
        uc: x-velocity on C-grid (input, output)
        vc: y-velocity on C-grid (input, output)
        vort_c: Vorticity on C-grid (input)
        ke_c: kinetic energy on C-grid (input)
        v: y-velocity on D-grid (input)
        u: x-velocity on D-grid (input)
        dt2: timestep (input)
    """
    # TODO: this function is kept because it has a translate test,
    # if the structure of call changes significantly from this
    # consider deleting this function and the translate test
    # or restructuring the savepoint
    c_sw._update_y_velocity(
        vort_c,
        ke_c,
        u,
        vc,
        c_sw.grid_data.cosa_v,
        c_sw.grid_data.sina_v,
        c_sw.grid_data.rdyc,
        dt2,
    )
    c_sw._update_x_velocity(
        vort_c,
        ke_c,
        v,
        uc,
        c_sw.grid_data.cosa_u,
        c_sw.grid_data.sina_u,
        c_sw.grid_data.rdxc,
        dt2,
    )


class TranslateC_SW(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        cgrid_shallow_water_lagrangian_dynamics = get_c_sw_instance(
            grid, namelist, stencil_factory
        )
        self.compute_func = cgrid_shallow_water_lagrangian_dynamics
        self.in_vars["data_vars"] = {
            "delp": {},
            "pt": {},
            "u": {"jend": grid.jed + 1},
            "v": {"iend": grid.ied + 1},
            "w": {},
            "uc": {"iend": grid.ied + 1},
            "vc": {"jend": grid.jed + 1},
            "ua": {},
            "va": {},
            "ut": {},
            "vt": {},
            "omga": {"serialname": "omgad"},
            "divgd": {"iend": grid.ied + 1, "jend": grid.jed + 1},
        }
        self.in_vars["parameters"] = ["dt2"]
        for name, info in self.in_vars["data_vars"].items():
            info["serialname"] = name + "d"
        self.out_vars = {}
        for v, d in self.in_vars["data_vars"].items():
            self.out_vars[v] = d
        for servar in ["delpcd", "ptcd"]:
            self.out_vars[servar] = {}
        # TODO: Fix edge_interpolate4 in d2a2c_vect to match closer and the
        # variables here should as well.
        self.max_error = 2e-10
        self.stencil_factory = stencil_factory

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        delpc, ptc = self.compute_func(**inputs)
        return self.slice_output(inputs, {"delpcd": delpc, "ptcd": ptc})


class TranslateDivergenceCorner(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.max_error = 9e-10
        self.cgrid_sw_lagrangian_dynamics = get_c_sw_instance(
            grid, namelist, stencil_factory
        )
        self.in_vars["data_vars"] = {
            "u": {
                "istart": grid.isd,
                "iend": grid.ied,
                "jstart": grid.jsd,
                "jend": grid.jed + 1,
            },
            "v": {
                "istart": grid.isd,
                "iend": grid.ied + 1,
                "jstart": grid.jsd,
                "jend": grid.jed,
            },
            "ua": {},
            "va": {},
            "divg_d": {},
        }
        self.out_vars = {
            "divg_d": {
                "istart": grid.isd,
                "iend": grid.ied + 1,
                "jstart": grid.jsd,
                "jend": grid.jed + 1,
            }
        }
        self.stencil_factory = stencil_factory

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        self.cgrid_sw_lagrangian_dynamics._divergence_corner(
            **inputs,
            dxc=self.grid.dxc,
            dyc=self.grid.dyc,
            sin_sg1=self.grid.sin_sg1,
            sin_sg2=self.grid.sin_sg2,
            sin_sg3=self.grid.sin_sg3,
            sin_sg4=self.grid.sin_sg4,
            cos_sg1=self.grid.cos_sg1,
            cos_sg2=self.grid.cos_sg2,
            cos_sg3=self.grid.cos_sg3,
            cos_sg4=self.grid.cos_sg4,
            rarea_c=self.grid.rarea_c,
        )
        return self.slice_output({"divg_d": inputs["divg_d"]})


class TranslateCirculation_Cgrid(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.max_error = 5e-9
        self.cgrid_sw_lagrangian_dynamics = get_c_sw_instance(
            grid, namelist, stencil_factory
        )
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "vort_c": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
            },
        }
        self.out_vars = {
            "vort_c": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
            }
        }
        self.stencil_factory = stencil_factory

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        self.cgrid_sw_lagrangian_dynamics._circulation_cgrid(
            **inputs,
            dxc=self.grid.dxc,
            dyc=self.grid.dyc,
        )
        return self.slice_output({"vort_c": inputs["vort_c"]})


class TranslateVorticityTransport_Cgrid(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        cgrid_sw_lagrangian_dynamics = get_c_sw_instance(
            grid, namelist, stencil_factory
        )

        def compute_func(*args, **kwargs):
            return compute_vorticitytransport_cgrid(
                cgrid_sw_lagrangian_dynamics, *args, **kwargs
            )

        self.compute_func = compute_func
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "vort_c": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
            },
            "ke_c": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
            },
            "u": {},
            "v": {},
        }
        self.in_vars["parameters"] = ["dt2"]
        self.out_vars = {"uc": grid.x3d_domain_dict(), "vc": grid.y3d_domain_dict()}
        self.stencil_factory = stencil_factory
