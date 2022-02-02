#!/bin/bash

JENKINS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

FV3GFS_IMAGE="driver_image" make -C ${JENKINS_DIR}/../docker fv3gfs_image
docker run --rm driver_image make -C /driver test test_mpi
make -C fv3gfs-physics get_test_data && docker run --rm --network host -v $(pwd)/test_data:/test_data driver_image mpirun -n 6 python3 -m mpi4py -m pace.driver.run /driver/examples/configs/baroclinic_c12_from_serialbox.yaml
