#!/bin/bash

# Change shell options to exist immediately if a command returns a non zero status
# and print trace of commands
set -ex

# Check number of arguments at call time
if [ "$#" -ne 7 ]; then
    echo "Usage: ./model_training.sh  project region bucket bucket_staging pipeline_version data_version runner"
    exit 1
fi

# Define global variables
PROJECT=$1
REGION=$2
BUCKET=$3
BUCKET_STAGING=$4
PIPELINE_VERSION=$5
DATA_VERSION=$6
MODEL_VERSION=$(date -u +%y%m%d_%H%M%S)
RUNNER=$7 # DirectRunner or AIplatformRunner to run on AI platform

TF_VERSION='2.2'
PYTHON_VERSION='3.7'

# Set up some globals for gcs file
HANDLER='gs://' # ../ for local data, gs:// for cloud data

BASE_DIR=${HANDLER}${BUCKET}/${PIPELINE_VERSION}
RUN_DIR=${BASE_DIR}/run/${DATA_VERSION}
DATA_DIR=${RUN_DIR}/data_transform
OUTPUT_DIR=${RUN_DIR}/model_training/${MODEL_VERSION}


JOB_NAME=${PIPELINE_VERSION}_${MODEL_VERSION}_hypertune
echo $OUTPUT_DIR $REGION $JOB_NAME

if [ "$RUNNER" = "DirectRunner" ]; then
    # Use AI Platform to train the model in local file system
    gcloud ai-platform local train \
        --module-name=trainer.task \
        --package-path=bikesharingmodel/trainer \
        -- \
        --data_dir=${DATA_DIR} \
        --output_dir=${OUTPUT_DIR}
else 
    echo "RUNNER = " $RUNNER
fi

if [ "$RUNNER" = "AIplatformRunner" ]; then
    gcloud ai-platform jobs submit training $JOB_NAME \
        --region=$REGION \
        --module-name=trainer.task \
        --package-path=./src/bikesharingmodel/trainer \
        --staging-bucket=gs://$BUCKET_STAGING \
        --runtime-version=$TF_VERSION \
        --python-version=$PYTHON_VERSION \
        --config=./src/hyperparam.yaml \
        --stream-logs \
        -- \
        --data_dir=$DATA_DIR \
        --output_dir=$OUTPUT_DIR \
        --num_epochs=1  
else 
    echo "RUNNER = " $RUNNER
fi

# Get information from the hyperparameter tuning job
echo "INFO: Extracting best hyperparameters from job $JOB_NAME"
TRIAL_ID=$(gcloud ai-platform jobs describe $JOB_NAME --format 'value(trainingOutput.trials.trialId.slice(0))')

# Write output file for next step in pipeline
echo $DATA_VERSION > /data_version.txt
echo $MODEL_VERSION > /model_version.txt
echo $TRIAL_ID > /trial_id.txt

# add tensorboard to artifacts
JOB_DIR=$OUTDIR/$TRIAL_ID 
metadata="{\"outputs\":[{\"type\":\"tensorboard\",\"source\":\"$JOB_DIR\"}]}"
echo $metadata > /mlpipeline-ui-metadata.json

echo "INFO: Done."