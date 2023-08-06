#!/bin/bash

tag=$(cat cellbender/__init__.py | sed -e 's?__version__ = ??' | sed "s/^'\(.*\)'$/\1/")
release=v${tag}

docker build \
    -t us.gcr.io/broad-dsde-methods/cellbender:${tag} \
    --build-arg GIT_SHA=${release} \
    -f docker/DockerfileGit \
    .
