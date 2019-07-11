# Start from nvidia-docker image with drivers pre-installed to use a GPU
FROM nvcr.io/nvidia/cuda:9.0-base-ubuntu16.04

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
 && curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && ~/miniconda/bin/conda install -y pytorch torchvision cudatoolkit=9.0 -c pytorch \
 && ~/miniconda/bin/conda install -y -c anaconda pytables \
 && ~/miniconda/bin/conda clean -ya \
 && git clone https://github.com/broadinstitute/CellBender.git cellbender \
 && yes | ~/miniconda/bin/pip install -e cellbender \
 && sudo rm -rf ~/.cache/pip

# Add cellbender command to PATH
ENV PATH="~/miniconda/bin:${PATH}"
