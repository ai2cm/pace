import mpi4py

import fv3gfs.wrapper as wrapper
from pace.util import io


# May need to run 'ulimit -s unlimited' before running this example
# If you're running in our prepared docker container, you definitely need to do this
# sets the stack size to unlimited

# Run using mpirun -n 6 python3 basic_model.py

if __name__ == "__main__":

    # MPI stuff
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()

    initial_names = [
        "specific_humidity",
        "cloud_water_mixing_ratio",
        "rain_mixing_ratio",
        "snow_mixing_ratio",
        "cloud_ice_mixing_ratio",
        "graupel_mixing_ratio",
        "ozone_mixing_ratio",
        "air_temperature",
        "pressure_thickness_of_atmospheric_layer",
        "vertical_thickness_of_atmospheric_layer",
        "logarithm_of_interface_pressure",
        "x_wind",
        "y_wind",
        "vertical_wind",
        "x_wind_on_c_grid",
        "y_wind_on_c_grid",
        "total_condensate_mixing_ratio",
        "interface_pressure",
        "surface_geopotential",
        "interface_pressure_raised_to_power_of_kappa",
        "surface_pressure",
        "vertical_pressure_velocity",
        "atmosphere_hybrid_a_coordinate",
        "atmosphere_hybrid_b_coordinate",
        "accumulated_x_mass_flux",
        "accumulated_y_mass_flux",
        "accumulated_x_courant_number",
        "accumulated_y_courant_number",
        "dissipation_estimate_from_heat_source",
        "eastward_wind",
        "northward_wind",
        "layer_mean_pressure_raised_to_power_of_kappa",
        "time",
    ]

    wrapper.initialize()
    state = wrapper.get_state(names=initial_names)
    io.write_state(state, "instate_{0}.nc".format(rank))
    for i in range(wrapper.get_step_count()):
        wrapper.step_dynamics()
        wrapper.step_physics()
        wrapper.save_intermediate_restart_if_enabled()
    state = wrapper.get_state(names=initial_names)
    io.write_state(state, "outstate_{0}.nc".format(rank))

    wrapper.cleanup()
