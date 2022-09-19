#!/bin/bash

echo "branch: $CIRCLE_BRANCH"
echo "tag:    $CIRCLE_TAG"

set -e
set -o pipefail

if [[ -z "$CIRCLE_SHA1" ]]
then
    CIRCLE_SHA1=$(git rev-parse HEAD)
fi

CACHE_IMAGE="us.gcr.io/vcm-ml/pace:latest"
BUILD_IMAGE="us.gcr.io/vcm-ml/pace:$CIRCLE_SHA1"

if [[ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]]
then
    echo "Google authentication not configured. "
    echo "Please set the GOOGLE_APPLICATION_CREDENTIALS environmental variable."
    exit 1
fi

echo $BUILD_IMAGE

BUILD_FLAGS=" \
    --secret id=gcp,src=$GOOGLE_APPLICATION_CREDENTIALS \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --progress=plain \
    --cache-from $CACHE_IMAGE \
"

PACE_IMAGE="$BUILD_IMAGE" DEV=n BUILD_FLAGS="$BUILD_FLAGS" make build

echo "pushing tagged images $CIRCLE_SHA1"
docker push $BUILD_IMAGE
docker tag $BUILD_IMAGE $CACHE_IMAGE
docker push $CACHE_IMAGE
