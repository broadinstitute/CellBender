# Start from nvidia-docker image with drivers pre-installed to use a GPU
FROM nvcr.io/nvidia/cuda:11.7.1-base-ubuntu18.04

# Specify a certain commit of CellBender via branch, tag, or sha
ARG GIT_SHA

LABEL maintainer="Stephen Fleming <sfleming@broadinstitute.org>"
ENV DOCKER=true \
    CONDA_AUTO_UPDATE_CONDA=false \
    CONDA_DIR="/opt/conda" \
    GCLOUD_DIR="/opt/gcloud" \
    GOOGLE_CLOUD_CLI_VERSION="397.0.0" \
    GIT_SHA=$GIT_SHA
ENV PATH="$CONDA_DIR/bin:$GCLOUD_DIR/google-cloud-sdk/bin:$PATH"

# Install miniconda and gsutil with compiled crcmod and pytorch and cellbender
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates sudo git \
 && apt-get clean \
 && sudo rm -rf /var/lib/apt/lists/* \
 && curl -so $HOME/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh \
 && chmod +x $HOME/miniconda.sh \
 && $HOME/miniconda.sh -b -p $CONDA_DIR \
 && rm $HOME/miniconda.sh

RUN mkdir -p $GCLOUD_DIR \
 && curl -so $HOME/google-cloud-cli.tar.gz https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-${GOOGLE_CLOUD_CLI_VERSION}-linux-x86_64.tar.gz \
 && tar -xzf $HOME/google-cloud-cli.tar.gz -C $GCLOUD_DIR \
 && pip install --no-cache-dir -U crcmod==1.7 \
 && .$GCLOUD_DIR/google-cloud-sdk/install.sh --usage-reporting false \
 && rm $HOME/google-cloud-cli.tar.gz \
 && sudo rm -rf ~/.cache/pip

# && conda install pytorch==1.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia \
RUN conda install pytorch==1.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge \
 && sudo rm -rf ~/.cache/pip \
 && conda clean -yaf \
 && mkdir -p /root/.cache/torch/kernels

RUN yes | pip install --no-cache-dir -U git+https://github.com/broadinstitute/CellBender.git@$GIT_SHA \
 && sudo rm -rf ~/.cache/pip
