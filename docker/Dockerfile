ARG serialbox_image

FROM $serialbox_image as fv3ser-environment

RUN apt-get update && apt-get install -y \
    llvm-9-dev \
    libclang-9-dev \
    libpython3-dev \
    python3 \
    python3-pip 
# if try to start directly with ubuntu: need at least cmake, libssl-dev, libboost-all-dev, git, but others as well    
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10

FROM fv3ser-environment as fv3ser-install
RUN pip install scikit-build
RUN git clone https://github.com/jdahm/dawn.git /usr/src/dawn
RUN cd /usr/src/dawn && git checkout 5d2939883eeaef8660479c42911f09b54647f5c3
RUN pip install /usr/src/dawn/dawn
RUN git clone --depth 1 -b field_dimensions_fix https://github.com/jdahm/gt4py /usr/src/gt4py
RUN pip install -e /usr/src/gt4py
RUN python /usr/src/gt4py/setup.py install_gt_sources --git-branch=v1.0.4
RUN pip install pytest==5.2.2 && pip install f90nml==1.1.0

# If we update to the latest gt4py, maybe 
# RUN python -m gt4py.gt_src_manager install

FROM fv3ser-install as fv3ser
COPY fv3 /fv3

