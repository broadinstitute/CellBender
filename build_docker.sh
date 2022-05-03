#!/bin/bash

version=$(<VERSION)

docker build -t cellbender:${version} -f docker/Dockerfile .