#!/bin/bash
PARENT=stablebaselines/stable-baselines3
TAG=stablebaselines/rl-baselines3-zoo
VERSION=0.10.0
read id
echo To use GPU = "True" to use CPU = "False"
read USE_GPU
echo $USE_GPU
echo $TAG
if [[ ${USE_GPU} == "True" ]]; then
    echo Using GPU
    PARENT="${PARENT}:${VERSION}"
else
    echo Using CPU
    PARENT="${PARENT}-cpu:${VERSION}"
    TAG="${TAG}-cpu"
fi

echo docker build --build-arg PARENT_IMAGE=${PARENT} --build-arg USE_GPU=${USE_GPU} -t ${TAG}:${VERSION}${id} . -f docker/Dockerfile
echo docker tag ${TAG}:${VERSION}${id} ${TAG}:latest

if [[ ${RELEASE} == "True" ]]; then
    docker push ${TAG}:${VERSION}
    docker push ${TAG}:latest
fi
