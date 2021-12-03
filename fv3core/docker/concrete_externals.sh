#!/usr/bin/env bash

command=$1

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

rm $SCRIPT_DIR/../external/pace-util
cp -r $SCRIPT_DIR/../../pace-util $SCRIPT_DIR/../external/pace-util

rm $SCRIPT_DIR/../external/stencils
cp -r $SCRIPT_DIR/../../stencils $SCRIPT_DIR/../external/stencils

echo $command
eval $command

ret=$?

rm -r $SCRIPT_DIR/../external/pace-util
rm -r $SCRIPT_DIR/../external/stencils

cd $SCRIPT_DIR
ln -s ../../pace-util ../external/pace-util
ln -s ../../stencils ../external/stencils

exit $ret
