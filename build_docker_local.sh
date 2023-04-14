#!/bin/bash

tag=$(<VERSION)

docker build \
    -t us.gcr.io/broad-dsde-methods/cellbender:${tag} \
    -f docker/Dockerfile \
    .
