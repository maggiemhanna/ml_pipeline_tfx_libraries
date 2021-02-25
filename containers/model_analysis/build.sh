#!/bin/bash -e

IMAGE_NAME=bikesharing-model-analysis

if [ -z "$1" ]; then
  PROJECT_ID=$(gcloud config config-helper --format "value(configuration.properties.core.project)")
else
  PROJECT_ID=$1
fi

if [ -z "$2" ]; then
  TAG_NAME=$(git describe --tags --always)
else
  TAG_NAME="$2"
fi

docker pull eu.gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest || true
docker build -t ${IMAGE_NAME} . --cache-from eu.gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest

docker tag ${IMAGE_NAME} eu.gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG_NAME}
docker tag ${IMAGE_NAME} eu.gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest
docker push eu.gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG_NAME}
docker push eu.gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest

# Output the strict image name (which contains the sha256 image digest)
docker inspect --format="{{index .RepoDigests 0}}" "eu.gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG_NAME}"