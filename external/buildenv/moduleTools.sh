#!/bin/bash

# module tools

##################################################
# functions
##################################################

exitError()
{
    \rm -f /tmp/tmp.${user}.$$ 1>/dev/null 2>/dev/null
    echo "ERROR $1: $3" 1>&2
    echo "ERROR     LOCATION=$0" 1>&2
    echo "ERROR     LINE=$2" 1>&2
    exit $1
}

showWarning()
{
    echo "WARNING $1: $3" 1>&2
    echo "WARNING       LOCATION=$0" 1>&2
    echo "WARNING       LINE=$2" 1>&2
}

containsElement()
{
  local e
  for e in "${@:2}"; do [[ "$e" == "$1" ]] && return 0; done
  return 1
}


isOnOff()
{
    local switch=$1
    local onoff=(ON OFF)
    containsElement "${switch}" "${onoff[@]}" || exitError 101 ${LINENO} "Invalid value for ON/OFF switch (${switch}) chosen"
}


checkModuleAvailable()
{
    local module=$1
    if [ -n "${module}" ] ; then
        module avail -t 2>&1 | grep "${module}" &> /dev/null
        if [ $? -ne 0 ] ; then
            exitError 201 ${LINENO} "module ${module} is unavailable"
        fi
    fi
}


compareFiles()
{
    one=$1
    two=$2
    msg=$3

    if [ ! -f "${one}" ] ; then exitError 3001 ${LINENO} "Must supply two valid files to compareFiles (${one})" ; fi
    if [ ! -f "${two}" ] ; then exitError 3002 ${LINENO} "Must supply two valid files to compareFiles (${two})" ; fi

    # sort and compare the two files
    diff <(sort "${one}") <(sort "${two}")

    if [ $? -ne 0 ] ; then
        echo "ERROR: Difference detected between ${one} and ${two} in compareFiles"
        echo "       ${msg}"
        exit 1
    fi

}


compilerVersion()
{
    compiler=$1

    # check for zero strings
    if [ -z "${compiler}" ] ; then exitError 3101 ${LINENO} "Must supply a compiler command to compilerVersion" ; fi

    # find absolute path of compiler
    which ${compiler} &> /dev/null
    if [ $? -eq 1 ] ; then exitError 3102 ${LINENO} "Cannot find compiler command (${compiler})" ; fi
    compiler=`which ${compiler}`

    # check for GNU
    res=`${compiler} -v 2>&1 | grep '^gcc'`
    if [ -n "${res}" ] ; then
        version=`echo "${res}" | awk '{print $3}'`
        echo ${version}
        return
    fi

    # check for Cray
    res=`${compiler} -V 2>&1 | grep '^Cray'`
    if [ -n "${res}" ] ; then
        version=`echo "${res}" | awk '{print $5}'`
        echo ${version}
        return
    fi

    # check for PGI
    res=`${compiler} -V 2>&1 | grep '^pg'`
    if [ -n "${res}" ] ; then
        version=`echo "${res}" | awk '{print $2}'`
        echo ${version}
        return
    fi

    # could not determine compiler version
    exitError 3112 ${LINENO} "Could not determine compiler version (${compiler})"

}


writeModuleList()
{
    local logfile=$1
    local mode=$2
    local msg=$3
    local modfile=$4

    # check arguments
    test -n "${logfile}" || exitError 601 ${LINENO} "Option <logfile> is not set"
    test -n "${mode}" || exitError 602 ${LINENO} "Option <mode> is not set"
    test -n "${msg}" || exitError 603 ${LINENO} "Option <msg> is not set"

    # check correct mode
    local modes=(all loaded)
    containsElement "${mode}" "${modes[@]}" || exitError 610 ${LINENO} "Invalid mode (${mode}) chosen"

    # clean log file for "all" mode
    if [ "${mode}" == "all" ] ; then
        /bin/rm -f ${logfile} 2>/dev/null
        touch ${logfile}
    fi

    # log modules to logfile
    echo "=============================================================================" >> ${logfile}
    echo "${msg}:" >> ${logfile}
    echo "=============================================================================" >> ${logfile}
    if [ "${mode}" == "all" ] ; then
        module avail -t >> ${logfile} 2>&1
    elif [ "${mode}" == "loaded" ] ; then
        module list -t 2>&1 | grep -v alps >> ${logfile}
    else
        exitError 620 ${LINENO} "Invalid mode (${mode}) chosen"
    fi

    # save list of loaded modules to environment file (if required)
    if [ -n "${modfile}" ] ; then
        /bin/rm -f ${modfile}
        touch ${modfile}
        module list -t 2>&1 | grep -v alps | grep -v '^- Package' | grep -v '^Currently Loaded' | sed 's/^/module load /g' > ${modfile}

        # Workaround for machines that store the modules in a predefined list
        # such as kesch
        if [[ -n "$ENVIRONMENT_TEMPFILE" ]] ; then
            cp $ENVIRONMENT_TEMPFILE ${modfile}
        else
            if [[ -z ${host} ]]; then
                exitError 654 ${LINENO} "host is not defined"
            fi
            # workaround for Todi, Daint, and Lema
            if [ "${host}" == "lema" -o "${host}" == "daint" -o "${host}" == "dom" ] ; then
                # Replace some module loads with a swap
                swap=( "gcc" "pgi" "cce" "cray-mpich" "cudatoolkit" )
                for i in "${swap[@]}"
                do
                    sed -i "s/module load ${i}/module swap ${i}/g" "${modfile}"
                done
                # Move the swap cudatoolkit to end of file
                cuda=$(grep cudatoolkit "${modfile}")
                if [ -n "${cuda}" ]; then
                    # Delete the line containing cuda
                    grep -v "${cuda}" ${modfile} > /tmp/tmp.${host}.${user}.$$
                    /bin/mv -f /tmp/tmp.${host}.${user}.$$ ${modfile}
                    # Add it to the end of the file
                    echo "${cuda}" >> ${modfile}
                fi
            fi
        fi
    fi
}


testEnvironment()
{
    local tmp=/tmp/tmp.${user}.$$

    echo ">>>>>>>>>>>>>>> test environment setup"

    # initialize the log
    writeModuleList ${tmp}.log all "AVAILABLE MODULES"

    # checkpoint environment before
    writeModuleList ${tmp}.log loaded "BEFORE C++ MODULES" ${tmp}.mod.before

    # change environments a couple of times
    for i in `seq 2` ; do

        echo ">>>>>>>>>>>>>>> test C++ environment setup"

        # check C++ env
        setCppEnvironment
        writeModuleList ${tmp}.log loaded "C++ MODULES" ${tmp}.mod.dycore
        if [ -z ${old_prgenv+x} ] ; then exitError 8001 ${LINENO} "variable old_prgenv is not set" ; fi
        if [ -z ${dycore_gpp+x} ] ; then exitError 8002 ${LINENO} "variable dycore_gpp is not set" ; fi
        if [ -z ${dycore_gcc+x} ] ; then exitError 8003 ${LINENO} "variable dycore_gcc is not set" ; fi
        if [ -z ${cuda_gpp+x} ] ; then exitError 8004 ${LINENO} "variable cuda_gpp is not set" ; fi
        if [ -z ${BOOST_PATH+x} ] ; then exitError 8005 ${LINENO} "variable BOOST_PATH is not set" ; fi

        # check cleanup of C++ env
        unsetCppEnvironment
        writeModuleList ${tmp}.log loaded "BETWEEN MODULES" ${tmp}.mod.between
        if [ ! -z ${old_prgenv+x} ] ; then exitError 8101 ${LINENO} "variable old_prgenv is still set" ; fi
        if [ ! -z ${dycore_gpp+x} ] ; then exitError 8102 ${LINENO} "variable dycore_gpp is still set" ; fi
        if [ ! -z ${dycore_gcc+x} ] ; then exitError 8103 ${LINENO} "variable dycore_gcc is still set" ; fi
        if [ ! -z ${cuda_gpp+x} ] ; then exitError 8104 ${LINENO} "variable cuda_gpp is still set" ; fi
        compareFiles ${tmp}.mod.before ${tmp}.mod.between

        echo ">>>>>>>>>>>>>>> test Fortran environment setup"

        # check Fortran env
        setFortranEnvironment
        writeModuleList ${tmp}.log loaded "FORTRAN MODULES" ${tmp}.mod.fortran
        if [ -z ${old_prgenv+x} ] ; then exitError 8201 ${LINENO} "variable old_prgenv is not set" ; fi

        # check cleanup of Fortran env
        unsetFortranEnvironment
        writeModuleList ${tmp}.log loaded "AFTER FORTRAN MODULES" ${tmp}.mod.after
        if [ ! -z ${old_prgenv+x} ] ; then exitError 8301 ${LINENO} "variable old_prgenv is still set" ; fi
        compareFiles ${tmp}.mod.before ${tmp}.mod.after

    done

    # everything ok
    echo ">>>>>>>>>>>>>>>   success"

    # remove temporary files
    /bin/rm -f ${tmp}*

}


writeCppEnvironment()
{
    setCppEnvironment
    writeModuleList /dev/null loaded "C++ MODULES" ${base_path}/modules_dycore.env
    unsetCppEnvironment
}


writeFortranEnvironment()
{
    setFortranEnvironment
    writeModuleList /dev/null loaded "FORTRAN MODULES" ${base_path}/modules_fortran.env
    unsetFortranEnvironment
}

export -f writeModuleList
export -f containsElement
