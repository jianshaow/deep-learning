FROM tensorflow/tensorflow:2.12.0-gpu

ARG HTTP_PROXY
ENV http_proxy=$HTTP_PROXY \
    https_proxy=$HTTP_PROXY \
    no_proxy=localhost,127.0.0.1

RUN apt-get update && apt-get install -y python3-tk graphviz && \
    useradd -ms /bin/bash devel
USER devel
RUN pip install --upgrade pip && pip install matplotlib pydot_ng
ENV HOME=/home/devel
ENV PYTHONPATH=$HOME/workspace
WORKDIR $HOME/workspace
