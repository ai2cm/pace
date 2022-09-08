The data in this directory is from here:
v1.0: gs://vcm-fv3config/data/initial_conditions/c12_baroclinic_restart_day_10/v1.0
v2.0: gs://vcm-fv3config/data/initial_conditions/gfs_c12_example/v1.0 
v3.0: gs://vcm-fv3config/data/initial_conditions/restart_initial_conditions/v1.0
v4.0: gs://vcm-fv3config/data/initial_conditions/c12_6ranks_standard_restart_15min/v1.0
v5.0: /home/ajdas/shield/regional_Laura/RESTART


example copy command:
gsutil -u vcm-ml cp -r "gs://vcm-fv3config/data/initial_conditions/restart_initial_conditions/v1.0/*" .