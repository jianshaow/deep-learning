FROM tensorflow/tensorflow:2.12.0-gpu

RUN apt-get update && apt-get install -y python3-tk graphviz && \
    useradd -ms /bin/bash devel
USER devel
RUN pip install --upgrade pip && pip install matplotlib pydot_ng
ENV HOME=/home/devel
ENV DISPLAY= WAYLAND_DISPLAY= XDG_RUNTIME_DIR= PULSE_SERVER= PYTHONPATH=$HOME/workspace
WORKDIR $HOME/workspace
