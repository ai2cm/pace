#!/bin/bash

IMAGE=us.gcr.io/vcm-ml/fv3core:0.1.0-wrapped

MOUNTS="-v $(pwd):/fv3core"

docker run --rm $MOUNTS -w /fv3core -it $IMAGE bash
