# Start from nvidia-docker image with drivers pre-installed to use a GPU
FROM nvcr.io/nvidia/cuda:11.3.1-base-ubuntu16.04

LABEL maintainer="Stephen Fleming <sfleming@broadinstitute.org>"
ENV DOCKER='true' \
    CONDA_AUTO_UPDATE_CONDA=false \
    PATH="/home/user/miniconda/bin:/root/google-cloud-sdk/bin:${PATH}"

# Add the local copy of cellbender to the docker image in /software
ADD . /software/cellbender

# TODO: consider installing gsutil like this to shrink image
## gsutil with compiled crcmod
#ENV GOOGLE_CLOUD_CLI_VERSION="397.0.0" \
#    PATH="${HOME}/google-cloud-sdk/bin:/software:${PATH}"
#RUN curl -so $HOME/google-cloud-cli.tar.gz https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-${GOOGLE_CLOUD_CLI_VERSION}-linux-x86_64.tar.gz \
# && tar -xzf $HOME/google-cloud-cli.tar.gz -C $HOME \
# && pip install --no-cache-dir -U crcmod==1.7 \
# && .$HOME/google-cloud-sdk/install.sh \
# && rm $HOME/google-cloud-cli.tar.gz \
# && rm -rf ~/.cache/pip

# Install miniconda and google cloud SDK and pytorch and cellbender
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates sudo \
 && apt-get clean \
 && sudo rm -rf /var/lib/apt/lists/* \
 && curl -so ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p /home/user/miniconda \
 && rm ~/miniconda.sh \
 && curl -sSL https://sdk.cloud.google.com | bash \
 && conda install -y pytorch==1.11.0 cudatoolkit=10.2 -c pytorch \
 && yes | pip install -e /software/cellbender \
 && sudo rm -rf ~/.cache/pip \
 && conda clean -yaf \
 && mkdir -p /root/.cache/torch/kernels
