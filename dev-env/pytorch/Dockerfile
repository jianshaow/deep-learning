ARG BASE_IMAGE=jianshao/dl-torch-dev
ARG TORCH_VER=2.2.1

FROM ${BASE_IMAGE}:${TORCH_VER}-${IMAGE_TYPE}

COPY requirements.txt .
ARG PYPI_INDEX_ARG=""
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir keras && \
    pip install --no-cache-dir -r requirements.txt ${PYPI_INDEX_ARG}
ENV KERAS_BACKEND=torch
