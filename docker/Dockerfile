FROM python:3.12-slim

LABEL maintainer="Stephen Fleming <sfleming@broadinstitute.org>"
ENV DOCKER=true \
    PATH="/opt/venv/bin:$PATH"

ARG VERSION

# procps (`ps`) is required for Seqera Fusion's in-container task supervisor -- without it,
# any Fusion-enabled run hard-fails immediately with "Command 'ps' required by nextflow to
# collect task metrics cannot be found", before the actual script ever runs.
RUN apt-get update && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends git curl gnupg procps \
 && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] \
    https://packages.cloud.google.com/apt cloud-sdk main" \
    > /etc/apt/sources.list.d/google-cloud-sdk.list \
 && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && pip install git+https://github.com/broadinstitute/CellBender@${VERSION} \
 && rm -rf ~/.cache/pip

# Deliberately no `ENTRYPOINT ["/bin/bash"]` (unlike upstream broadinstitute/CellBender's own
# Dockerfile, which this fork otherwise mirrors) -- Seqera Wave's Fusion-wrapping rewrites a
# container's entrypoint to bootstrap Fusion, and its handoff logic breaks when the ORIGINAL
# entrypoint already is a bare shell: every AWS Batch/Fusion task failed at exec with
# "/usr/bin/bash: /usr/bin/bash: cannot execute binary file" (exit 126) -- bash's own message
# for "asked to source a binary file as a script", not a kernel exec failure. Confirmed via
# controlled A/B testing (same launch config, only this ENTRYPOINT line changed) that dropping
# it to a bare CMD fixes it, independent of base OS/glibc version (Ubuntu 24.04 and Debian 13
# "trixie" both reproduced the bug WITH the explicit ENTRYPOINT, and both worked cleanly
# without it). Full writeup: docs/cellbender-fusion-entrypoint-bug.md in srt-tx/nf-sc-analysis.
CMD ["bash"]
