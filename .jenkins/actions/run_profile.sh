#!/bin/bash

set -e -x

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
ROOT_DIR="$(dirname "$(dirname "$SCRIPTPATH")")"
TESTDATA_PATH="/project/s1053/fv3core_serialized_test_data"
FORTRAN_SERIALIZED_DATA_VERSION=7.2.5

test -n "${experiment}" || exitError 1001 ${LINENO} "experiment is not defined"
test -n "${backend}" || exitError 1002 ${LINENO} "backend is not defined"

data_path=${TESTDATA_PATH}/${FORTRAN_SERIALIZED_DATA_VERSION}/${experiment}

$ROOT_DIR/examples/standalone/benchmarks/run_on_daint.sh 2 6 $backend "" $data_path "" "--profile"

cp $ROOT_DIR/fv3core_${experiment}_${backend}*.prof /project/s1053/performance/fv3core_monitor/$backend/

rm -rf .gt_cache_0000*

# generate simple profile listing
source $ROOT_DIR/venv/bin/activate
cat > $ROOT_DIR/profile.py <<EOF
#!/usr/bin/env python3

import pstats

stats = pstats.Stats("$ROOT_DIR/fv3core_${experiment}_${backend}_0.prof")
stats.strip_dirs()
stats.sort_stats('cumulative')
stats.print_stats(200)
print('=================================================================')
stats.sort_stats('calls')
stats.print_stats(200)
EOF
chmod 755 $ROOT_DIR/profile.py
$ROOT_DIR/profile.py > profile.txt

# convert to html
mkdir -p html
echo "<html><body><pre>" > html/index.html
cat profile.txt >> html/index.html
echo "</pre></body></html>" >> html/index.html

# so long!
exit 0
