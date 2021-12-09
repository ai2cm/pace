envdir="${envloc}/env"
if [ -d "${envdir}" ] ; then
    pushd "${envdir}" > /dev/null
        `git pull &>/dev/null`
        if [ $? -ne 0 ] ; then
            echo "WARNING: Problem pulling the buildenv. Defaulting to offline mode."
        fi
    popd
else
    git clone https://github.com/ai2cm/buildenv ${envdir} &>/dev/null
    if [ $? -ne 0 ] ; then
        echo "Error: Could not download the buildenv (https://github.com/ai2cm/buildenv) into ${envdir}. Aborting."
        exit 1
    else
        cd buildenv && git checkout feature/consolidate_daint_venv
    fi
fi
