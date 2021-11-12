ARG FORTRAN_IMAGE
ARG FMS_IMAGE
ARG ESMF_IMAGE
ARG SERIALBOX_IMAGE
ARG ENVIRONMENT_IMAGE

FROM $FORTRAN_IMAGE as fortran_image
FROM $ESMF_IMAGE as esmf_image
FROM $FMS_IMAGE as fms_image
FROM $SERIALBOX_IMAGE as serialbox_image
FROM $ENVIRONMENT_IMAGE as environment

FROM environment as fv3core

RUN pip3 install --upgrade pip setuptools wheel && \
    ln -s /bin/python3 /bin/python && \
    ln -s /bin/pip3 /bin/pip

COPY constraints.txt requirements.txt  /
COPY requirements  /requirements
RUN pip3 install -r /requirements.txt -c /constraints.txt

COPY --from=serialbox_image /usr/local/serialbox /usr/local/serialbox
COPY --from=serialbox_image /usr/include/boost /usr/include/boost

###
### Build and install GT4Py
###
ENV BOOST_HOME=/usr/include/boost
ARG CPPFLAGS="-I${BOOST_HOME} -I${BOOST_HOME}/boost"
ARG GT4PY_OPTIONALS=""
ARG GT4PY_DIR
COPY ${GT4PY_DIR} /gt4py

RUN if [ ! -z `echo $GT4PY_OPTIONALS | grep cuda` ] ; then pip install cupy-cuda102==7.7.0 ; else echo Not installing cuda ; fi
RUN pip install --no-cache-dir -c /constraints.txt "/gt4py${GT4PY_OPTIONALS}" && \
    python3 -m gt4py.gt_src_manager install

# This environment flag sets rebuild=False in gtscript.stencil calls
ENV FV3_STENCIL_REBUILD_FLAG=False

###
### Copy over necessary source and configuration files
###
COPY external/fv3gfs-util/ /external/fv3gfs-util/
COPY fv3core /fv3core/fv3core
COPY tests /fv3core/tests
COPY setup.py setup.cfg README.md /fv3core/
COPY docker/entrypoint.sh /entrypoint.sh

# Docker hard limits shared memory usage. MPICH for oversubscribed situation
# uses shared mem for most of its comunication operations,
# which leads to a sigbus crash.
# Both of those (for version <3.2 and >3.2) will force mpich to go
# through the network stack instead of using the shared nemory
# The cost is a slower runtime
ENV MPIR_CVAR_NOLOCAL=1
ENV MPIR_CVAR_CH3_NOLOCAL=1

RUN chmod +x /entrypoint.sh && \
    /entrypoint.sh

ARG ENV_CUDA_PATH=""
ENV CUDA_PATH=${ENV_CUDA_PATH}
ENV IN_DOCKER=True

FROM fv3core AS fv3core_wrapper

COPY --from=fortran_image /NCEPlibs /NCEPlibs

# install NCEP libraries
RUN cd /NCEPlibs && \
    mkdir /opt/NCEPlibs && \
    echo "y" | ./make_ncep_libs.sh -s linux -c gnu -d /opt/NCEPlibs -o 1

ENV FMS_DIR=/FMS \
    ESMF_DIR=/usr/local/esmf \
    MPI=mpich

ENV ESMF_INC="-I${ESMF_DIR}/include" \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ESMF_DIR}/lib:${FMS_DIR}/libFMS/.libs/

COPY --from=fms_image /FMS $FMS_DIR
COPY --from=esmf_image /usr/local/esmf $ESMF_DIR
COPY --from=fortran_image /FV3/ /external/fv3gfs-fortran/FV3/
COPY --from=fortran_image /stochastic_physics/ /external/fv3gfs-fortran/stochastic_physics/

COPY requirements/requirements_wrapper.txt /
RUN pip3 install -r /requirements_wrapper.txt -c /constraints.txt

COPY external/fv3gfs-wrapper /external/fv3gfs-wrapper

RUN /entrypoint.sh
