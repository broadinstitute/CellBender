# Start from nvidia-docker image with CUDA drivers pre-installed to use a GPU
FROM nvcr.io/nvidia/cuda:12.1.0-base-ubuntu22.04

# Copy the local cellbender repo
ADD . /software/cellbender

LABEL maintainer="Stephen Fleming <sfleming@broadinstitute.org>"
ENV DOCKER=true \
    DEBIAN_FRONTEND=noninteractive \
    GCLOUD_DIR="/opt/gcloud" \
    GOOGLE_CLOUD_CLI_VERSION="473.0.0" \
    GIT_SHA=$GIT_SHA
ENV PATH="$GCLOUD_DIR/google-cloud-sdk/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates sudo gnupg software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
# bootstrap pip for Python 3.12
 && python3.12 -m ensurepip --upgrade \
 && python3.12 -m pip install --upgrade pip \
# set python3.12 as default
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
 && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
# get gcloud CLI
 && mkdir -p $GCLOUD_DIR \
 && curl -so /tmp/google-cloud-cli.tar.gz \
    https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-${GOOGLE_CLOUD_CLI_VERSION}-linux-x86_64.tar.gz \
 && tar -xzf /tmp/google-cloud-cli.tar.gz -C $GCLOUD_DIR \
 && $GCLOUD_DIR/google-cloud-sdk/install.sh --usage-reporting false --quiet \
 && rm /tmp/google-cloud-cli.tar.gz \
# install cellbender and its dependencies
 && pip install /software/cellbender/ \
 && rm -rf ~/.cache/pip
