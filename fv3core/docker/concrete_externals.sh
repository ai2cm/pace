#!/usr/bin/env bash

command=$1

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

rm $SCRIPT_DIR/../external/fv3gfs-util
cp -r $SCRIPT_DIR/../../fv3gfs-util $SCRIPT_DIR/../external/fv3gfs-util

echo $command
eval $command

ret=$?

rm -r $SCRIPT_DIR/../external/fv3gfs-util
cd $SCRIPT_DIR
ln -s ../../fv3gfs-util ../external/fv3gfs-util

exit $ret
