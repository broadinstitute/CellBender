FROM python:3.12-slim

LABEL maintainer="Stephen Fleming <sfleming@broadinstitute.org>"
ENV DOCKER=true \
    PATH="/opt/venv/bin:$PATH"

ARG VERSION

RUN apt-get update && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends git curl gnupg \
 && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
 && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] \
    https://packages.cloud.google.com/apt cloud-sdk main" \
    > /etc/apt/sources.list.d/google-cloud-sdk.list \
 && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && pip install git+https://github.com/broadinstitute/CellBender@${VERSION} \
 && rm -rf ~/.cache/pip
