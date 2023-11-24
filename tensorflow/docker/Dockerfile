FROM tensorflow/tensorflow:2.14.0-gpu

COPY deep-learning.packages.txt /deep-learning.packages.txt
RUN /setup.packages.sh /deep-learning.packages.txt && \
    useradd -ms /bin/bash devel
USER devel
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir matplotlib pydot_ng
ENV HOME=/home/devel
ENV PYTHONPATH=$HOME/workspace
WORKDIR $HOME/workspace
