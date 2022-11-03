# Driver Configurations
Currently, driver takes in a yaml config containing the following options. Example configuration file can be found in examples/configs.

## stencil_config
Configuration options pertaining to GT4Py stencils such as backend and whether to rebuild stencils.

## initialization_type
Driver initialization type can be either `baroclinic` or `serialbox`.

## initialization_config
**`baroclinic`**:  no additional config needs to be specified.

**`serialbox`**:  `path` to the test data directory and optionally `serialized_grid` to specify whether to create grid from serialized data or calculate them in Python. The directory should have the necessary serialized data and a fortran namelist which will be used to set options such as dycore_config and physics_config. Currently, the supported serialized dataset is the same as the test case in pace-physics. To obtain this data, run the following command at the top level:
```
make -C physics get_test_data
```
[TODO] Update this once driver specific savepoints are added.

## performance_config
Configuration for doing performance timing. \
timer: true if performance timing is desired \
json_dump: true if timing output should be saved as a json file \
experiment_name: default to test if not specified

## diagnostics_config
Diagnostics configuration specifies where and what variables to save from the model. Currently, we support writing to `zarr` format.

## dycore_config
Dynamical core configuration \
To override this with a Fortran nml file, use `namelist_override`. See serialbox example yaml for more information.

## physics_config
Physics configuration

## dt_atmos*
Time step in seconds

## days/hours/minutes/seconds*
Simulation runtime \
\
*Note that this is required for serialbox initialization, time information in the nml file will not be used.

## ----**The following options need to match .nml file when using namelist_override**----
## nx_tile
Number of grid cells in the x-direction on one tile of the domain. This number is then duplicated in the y-direction since we only support `nx = ny`.

## nz
Number of vertical levels

## layout
Processor layout on each tile
