# Physics
Python implementation of GFS physics built using the GT4Py domain-specific language in Python.

## Description
pace-physics currently includes GFDL cloud microphysics scheme. Additional physics schemes (NOAH land surface, GFS sea ice, scale-aware mass-flux shallow convection, hybrid eddy-diffusivity mass-flux PBL and free atmospheric turbulence, and rapid radiative transfer model) have been ported indendepently and are available in the [physics standalone](https://github.com/ai2cm/physics_standalone) repository, additional work is required to integrate these schemes.

## Development guidelines

State container classes are highly recommended for use with physics schemes because only certain variables interacts with rest of the model, it is easier to follow variables shared across physics schemes, and it gives easy access to variable documentation. For example, microphysics state is the only state input into microphysics:

```python
physics_state = PhysicsState(...)
microphysics = Microphysics(stencil_factory, grid_data, namelist)
# make microphysics state from physics state
microphysics(physics_state.microphysics, timestep)
```

See `physics_state.py` for more information.
