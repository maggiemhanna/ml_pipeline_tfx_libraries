#!/bin/bash

# Change shell options to exist immediately if a command returns a non zero status
# and print trace of commands
set -ex

# Check number of arguments at call time
if [ "$#" -ne 8 ]; then
    echo "Usage: ./model_deployment.sh  project region bucket pipeline_version data_version model_version trial_id deployment_flag"
    exit 1
fi

# Define global variables
PROJECT=$1
REGION=$2
BUCKET=$3
PIPELINE_VERSION=$4
DATA_VERSION=$5
MODEL_VERSION=$6
TRIAL_ID=$7
DEPLOYMENT_FLAG=$8

TF_VERSION='2.2'
PYTHON_VERSION='3.7'

# Activate service account
gcloud auth activate-service-account --key-file /secret/gcp-credentials/user-gcp-sa.json

# Set project
gcloud config set project $PROJECT

# Set up some globals for gcs file
HANDLER='gs://' # ../ for local data, gs:// for cloud data

BASE_DIR=${HANDLER}${BUCKET}/${PIPELINE_VERSION}
RUN_DIR=${BASE_DIR}/run/${DATA_VERSION}
MODEL_DIR=${RUN_DIR}/model_training/${MODEL_VERSION}/${TRIAL_ID}/export/exporter

MODEL_NAME="bike_sharing_model"
MODEL_VERSION_NAME=${PIPELINE_VERSION}__${DATA_VERSION}__${MODEL_VERSION}

for EXPORTER in $(gsutil ls  ${MODEL_DIR}); do 
    EXPORTER_DIR=$EXPORTER
done; 

# Deploy best model from hyperparameter tuning
if [ $DEPLOYMENT_FLAG == 'OK' ]; then
    if gcloud ai-platform models create ${MODEL_NAME} --regions ${REGION}; then
        echo "INFO: A new model ${MODEL_NAME} has been created on AI Platform."
    else
        echo "INFO: A model with same name already exists on AI Platform."
    fi
    if gcloud ai-platform versions create ${MODEL_VERSION_NAME} --model ${MODEL_NAME} --origin ${EXPORTER_DIR} --runtime-version ${TF_VERSION} --python-version ${PYTHON_VERSION}; then
        echo "INFO: A new version ${MODEL_VERSION_NAME} has been deployed for model ${MODEL_NAME}."
        gcloud ai-platform versions set-default ${MODEL_VERSION_NAME} --model=${MODEL_NAME} 
        echo "INFO: The new version ${MODEL_VERSION_NAME} has been set as default version for model ${MODEL_NAME}."
    
    # Update pipeline frozen statistics and schema
    DATA_SCHEMA_DIR=${RUN_DIR}/data_validation/schema/data_schema.txt
    FREEZE_SCHEMA_DIR=gs://${BUCKET}/${PIPELINE_VERSION}/freeze/frozen_schema.txt

    # Update pipeline frozen statistics and schema
    DATA_STATS_DIR=${RUN_DIR}/data_validation/stats/data_stats.txt
    FREEZE_STATS_DIR=gs://${BUCKET}/${PIPELINE_VERSION}/freeze/frozen_stats.txt

    gsutil cp ${DATA_SCHEMA_DIR} ${FREEZE_SCHEMA_DIR}
    gsutil cp ${DATA_STATS_DIR} ${FREEZE_STATS_DIR}
    
    echo "INFO: Pipeline frozen statistics and schema were updated."

    else
        echo "INFO: A version with same name exists. This version has already been deployed and will not be deployed again."
    fi

else
    echo "INFO: The new model does not improve performance and was not deployed."
fi

echo "INFO: Done."