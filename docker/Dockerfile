# Start from nvidia-docker image with drivers pre-installed to use a GPU
FROM nvcr.io/nvidia/cuda:9.2-base-ubuntu16.04

LABEL maintainer="Stephen Fleming <sfleming@broadinstitute.org>"

# Install curl and sudo and git and miniconda and pytorch, cudatoolkit, pytables, and cellbender
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    sudo \
 && sudo apt-get install -y --no-install-recommends \
    git \
    bzip2 \
 && sudo rm -rf /var/lib/apt/lists/* \
 && curl -so ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p /home/user/miniconda \
 && rm ~/miniconda.sh

ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y pytorch torchvision cudatoolkit=9.2 -c pytorch \
 && conda install -y -c anaconda pytables \
 && conda clean -ya

ENV DOCKER='true'

RUN git clone https://github.com/broadinstitute/CellBender.git cellbender \
 && yes | pip install -e cellbender \
 && sudo rm -rf ~/.cache/pip

# google cloud SDK and gsutil
RUN curl -sSL https://sdk.cloud.google.com | bash
RUN conda install -y -c conda-forge google-api-python-client oauth2client google-auth \
 google-cloud-sdk google-auth-oauthlib \
 && conda clean -ya

# Add cellbender command to PATH
ENV PATH="~/miniconda/bin:${PATH}"
