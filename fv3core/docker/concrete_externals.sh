#!/usr/bin/env bash

command=$1

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

rm $SCRIPT_DIR/../external/pace-util
cp -r $SCRIPT_DIR/../../pace-util $SCRIPT_DIR/../external/pace-util

rm $SCRIPT_DIR/../external/stencils
cp -r $SCRIPT_DIR/../../stencils $SCRIPT_DIR/../external/stencils

rm $SCRIPT_DIR/../external/gt4py
cp -r $SCRIPT_DIR/../../external/gt4py $SCRIPT_DIR/../external/gt4py

echo $command
eval $command

ret=$?

rm -r $SCRIPT_DIR/../external/pace-util
rm -r $SCRIPT_DIR/../external/stencils
rm -r $SCRIPT_DIR/../external/gt4py

cd $SCRIPT_DIR
ln -s ../../pace-util ../external/pace-util
ln -s ../../stencils ../external/stencils
ln -s ../../stencils ../external/gt4py

exit $ret
