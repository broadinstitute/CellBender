# Start from nvidia-docker image with drivers pre-installed to use a GPU
FROM nvcr.io/nvidia/cuda:11.3.1-base-ubuntu16.04

LABEL maintainer="Stephen Fleming <sfleming@broadinstitute.org>"
ENV DOCKER='true'
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH="/home/user/miniconda/bin:/root/google-cloud-sdk/bin:${PATH}"

# Add the local copy of cellbender to the docker image in /software
ADD . /software/cellbender

# Install miniconda and google cloud SDK and pytorch and cellbender
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates sudo \
 && sudo rm -rf /var/lib/apt/lists/* \
 && curl -so ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p /home/user/miniconda \
 && rm ~/miniconda.sh \
 && curl -sSL https://sdk.cloud.google.com | bash \
 && conda install -y pytorch==1.11.0 cudatoolkit=10.2 -c pytorch \
 && yes | pip install -e /software/cellbender \
 && sudo rm -rf ~/.cache/pip \
 && conda clean -ya \
 && mkdir -p /root/.cache/torch/kernels