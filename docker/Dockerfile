ARG build_image
FROM $build_image as fv3ser

COPY external/fv3util/requirements.txt /usr/src/fv3util/requirements.txt
RUN pip install --no-cache-dir pytest==5.2.2 f90nml==1.1.0 pytest-subtests==0.3.0 pytest-xdist pytest-dependency scipy mpi4py && \
    pip install -r /usr/src/fv3util/requirements.txt
COPY external/fv3util /usr/src/fv3util
RUN pip install --no-cache-dir -e /usr/src/fv3util
COPY fv3core /fv3core/fv3core
COPY tests /fv3core/tests
COPY .git /fv3core/.git
COPY setup.py setup.cfg requirements.txt HISTORY.md README.md /fv3core/
RUN cd /fv3core && \
    git clean -fXd && \
    rm -rf /fv3core/.git && \
    pip install -e /fv3core 

