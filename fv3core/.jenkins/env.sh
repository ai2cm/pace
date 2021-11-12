envdir="${envloc}/env"
if [ -d "${envdir}" ] ; then
    pushd "${envdir}" > /dev/null
        `git pull &>/dev/null`
        if [ $? -ne 0 ] ; then
            echo "WARNING: Problem pulling the buildenv. Defaulting to offline mode."
        fi
    popd
else
    git clone https://github.com/VulcanClimateModeling/buildenv ${envdir} &>/dev/null
    if [ $? -ne 0 ] ; then
        echo "Error: Could not download the buildenv (https://github.com/VulcanClimateModeling/buildenv) into ${envdir}. Aborting."
        exit 1
    fi
fi
