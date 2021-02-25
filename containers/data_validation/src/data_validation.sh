#!/bin/bash

# Change shell options to exist immediately if a command returns a non zero status
# and print trace of commands
set -ex

# Check number of arguments at call time
if [ "$#" -ne 6 ]; then
    echo "Usage: ./data_validation.sh project region raw_data_path \
     bucket pipeline_version runner"
    exit 1
fi

PROJECT=$1
REGION=$2
RAW_DATA_PATH=$3
BUCKET=$4
PIPELINE_VERSION=$5
RUNNER=$6 # DirectRunner or DataflowRunner
DATA_VERSION=$(date -u +%y%m%d_%H%M%S)

# To run TFDV on Google Cloud, the TFDV wheel file must be downloaded and provided to the Dataflow workers. We can download the wheel file to the current directory as follows:

pip download tensorflow_data_validation \
  --no-deps \
  --platform manylinux1_x86_64 \
  --only-binary=:all:
  
# Callthe data validation job
python ./src/data_validation.py \
    --project=${PROJECT} \
    --region=${REGION} \
    --raw_data_path=${RAW_DATA_PATH} \
    --bucket=${BUCKET} \
    --pipeline_version=${PIPELINE_VERSION} \
    --data_version=${DATA_VERSION} \
    --runner=${RUNNER}