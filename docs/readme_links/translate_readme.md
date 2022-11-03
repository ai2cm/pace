# Translate tests
Translate tests are used to validate individual components of the model. There are two types of tests: sequential and parallel. Parallel tests are required when halo exchanges are present.

## Physics translate tests

### Serialized data infrastructure

Serialized data for the physics translate tests have a varierty of formats. By default, translate physics expects data in the collapsed format `[i*j, k]` with reversed k-axis (`k=0` is the surface). Read physics serialized data then reorders it to `[i, j, k]` assuming C-like index order and change `k=0` to be the top of the atmosphere. Additional formatting includes:
1. `dycore: True`. Default is False.
Input variable has the same format as dynamical core variables, revert to `read_serialized_data` in `TranslateFortranData2Py`.
2. `microph: True`. Default is False. Input variable has the format [i*j, 1, k] with the same k-axis as dynamical core.
3. `dwind: True`. Default is False. Special care is required as some grid variables are not currently supported.
4. `order: F`. Default is "C". By default the reorder assumes C-like index, in some cases it is required to use Fortran-like index order in order to match validation
5. `in_roll_zero: True` or `out_roll_zero: True`. Default is False. Some Fortran data pads zero in the opposite location as dycore (front vs. back), this option rolls the padding to the other direction for either the input or output (computed from input) data
6. `manual: True`. Default is False. For some output variables, they cannot go through `slice_output` because serialized output expects a different format. This option allows it to be skipped during `slice_output` and can be maually added back in afterwards.

### Savepoint tests

#### Serialized tests
- AtmosPhysDriverStatein: performs a mass adjustment to be consisent with GFS physics
- FillGFS: perform vertical filling to fix negative humidity and limit values to a minimum of 1e-9
- GFSPhysicsDriver: this is the physics driver call. **Warning:** This needs to be more modular if we want to test different physics combination
- Microph: GFDL cloud microphysics
- PhiFV3: stencil to adjust the height-z hydrostatically
- PrsFV3: stencil to adjust the geopotential height hydrostatically
- UpdateDWindsPhys: A-grid to D-grid wind
- PhysUpdatePressureSurfaceWinds: update pressure values and surface winds
- PhysUpdateTracers: gather tendencies and adjust dycore tracers values

#### Parallel tests
- FVUpdatePhys: apply physics tendencies consistent with the FV3 discretization and definition
