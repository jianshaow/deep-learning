ARG BASE_IMAGE=tensorflow/tensorflow
ARG TF_VER=2.16.1

FROM ${BASE_IMAGE}:${TF_VER}

COPY packages.txt ./
RUN ./setup.packages.sh packages.txt && \
    useradd -ms /bin/bash devel
USER devel
ENV HOME=/home/devel
ENV PATH=$HOME/.local/bin:$PATH
WORKDIR $HOME

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir matplotlib pydot_ng imageio
