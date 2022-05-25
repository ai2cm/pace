#!/usr/bin/env bash

command=$1

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

function cleanup {
  echo "Restoring fv3core symlinks"
  rm -rf $SCRIPT_DIR/../external/pace-util
  rm -rf $SCRIPT_DIR/../external/stencils
  rm -rf $SCRIPT_DIR/../external/gt4py
  rm -rf $SCRIPT_DIR/../external/dsl
  rm $SCRIPT_DIR/../constraints.txt

  cd $SCRIPT_DIR
  ln -s ../../pace-util ../external/pace-util
  ln -s ../../stencils ../external/stencils
  ln -s ../../external/gt4py ../external/gt4py
  ln -s ../../dsl ../external/dsl
  cd $SCRIPT_DIR/..
  ln -s ../constraints.txt constraints.txt
}

trap cleanup EXIT

echo "Replacing fv3core symlinks with concrete files"

rm $SCRIPT_DIR/../external/pace-util
cp -r $SCRIPT_DIR/../../pace-util $SCRIPT_DIR/../external/pace-util

rm $SCRIPT_DIR/../external/stencils
cp -r $SCRIPT_DIR/../../stencils $SCRIPT_DIR/../external/stencils

rm $SCRIPT_DIR/../external/gt4py
cp -r $SCRIPT_DIR/../../external/gt4py $SCRIPT_DIR/../external/gt4py

rm $SCRIPT_DIR/../external/dsl
cp -r $SCRIPT_DIR/../../dsl $SCRIPT_DIR/../external/dsl

rm $SCRIPT_DIR/../constraints.txt
cp $SCRIPT_DIR/../../constraints.txt $SCRIPT_DIR/../constraints.txt

echo $command
eval $command

ret=$?

exit $ret
