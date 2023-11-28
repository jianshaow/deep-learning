FROM tensorflow/tensorflow:2.15.0-gpu

COPY packages.txt /
RUN /setup.packages.sh /packages.txt && \
    useradd -ms /bin/bash devel
USER devel
ENV HOME=/home/devel
ENV PATH=$HOME/.local/bin:$PATH
ENV PYTHONPATH=$HOME/workspace
WORKDIR $HOME/workspace

RUN pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir matplotlib pydot_ng -i https://pypi.tuna.tsinghua.edu.cn/simple
