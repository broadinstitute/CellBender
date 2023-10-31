#!/bin/bash

tag=$(cat cellbender/VERSION.txt)
release=v${tag}

docker build \
    -t us.gcr.io/broad-dsde-methods/cellbender:${tag} \
    --build-arg GIT_SHA=${release} \
    -f docker/DockerfileGit \
    .
