import fv3config
import yaml

fv3config.ensure_data_is_downloaded()
config = yaml.safe_load(open("fv3config.yml", "r"))
fv3config.write_run_directory(config, "/rundir")
