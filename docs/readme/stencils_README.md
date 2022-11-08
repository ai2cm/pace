# Stencils
This package includes stencils shared across driver, fv3core, and physics.

## Description
List of classes in stencils:
- [`CopyCorners`](https://github.com/ai2cm/pace/blob/97018f21a820696b7e9fd5d8d3428b702819b834/stencils/pace/stencils/corners.py#L17)
- [`FillCorners`](https://github.com/ai2cm/pace/blob/97018f21a820696b7e9fd5d8d3428b702819b834/stencils/pace/stencils/corners.py#L545)
- [`CubedToLatLon`](https://github.com/ai2cm/pace/blob/97018f21a820696b7e9fd5d8d3428b702819b834/stencils/pace/stencils/c2l_ord.py#L100)
- [`ApplyPhysicsToDycore`](https://github.com/ai2cm/pace/blob/97018f21a820696b7e9fd5d8d3428b702819b834/stencils/pace/stencils/fv_update_phys.py#L77)
- [`DycoreToPhysics`](https://github.com/ai2cm/pace/blob/97018f21a820696b7e9fd5d8d3428b702819b834/stencils/pace/stencils/update_atmos_state.py#L148)
- [`UpdateAtmosphereState`](https://github.com/ai2cm/pace/blob/97018f21a820696b7e9fd5d8d3428b702819b834/stencils/pace/stencils/update_atmos_state.py#L235)
- [`AGrid2DGridPhysics`](https://github.com/ai2cm/pace/blob/97018f21a820696b7e9fd5d8d3428b702819b834/stencils/pace/stencils/update_dwind_phys.py#L152)

## Testing
This folder contains translate test infrastructure, where the regression data are generated from the [fv3gfs-fortran](https://github.com/ai2cm/fv3gfs-fortran) repository with serialization statements and a build procedure defined in [tests/serialized_test_data_generation](https://github.com/ai2cm/fv3gfs-fortran/tree/master/tests/serialized_test_data_generation). The regression data are generated daily and stored in Google Cloud Bucket.

### Running translate tests

#### Docker
```shell
cd $(git rev-parse --show-toplevel)
DEV=y USE_FTP=yes make savepoint_tests
```

#### Bare-metal
```shell
cd $(git rev-parse --show-toplevel)/fv3core
make USE_FTP=yes get_test_data
cd $(git rev-parse --show-toplevel)
pytest -v -s --data_path=fv3core/test_data/8.1.3/c12_6ranks_standard/dycore/ fv3core/tests
mpirun -np 6 python -m mpi4py -m pytest -v -s -m parallel --data_path=fv3core/test_data/8.1.3/c12_6ranks_standard/dycore fv3core/tests
```
