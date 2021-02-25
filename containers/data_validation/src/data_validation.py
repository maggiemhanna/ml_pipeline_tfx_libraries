#!/usr/bin/env python3

# Importing Librarires

import argparse
from datetime import datetime

import pkg_resources
import json
import sys
import os
import re

import tensorflow as tf
import tensorflow_data_validation as tfdv
from tensorflow_data_validation import StatsOptions

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow.python.lib.io import file_io
from apache_beam.options.pipeline_options import (
    PipelineOptions,
    GoogleCloudOptions,
    StandardOptions,
    SetupOptions,
    WorkerOptions
)

from data_validation_utils import *

print('INFO: TF version -- {}'.format(tf.__version__))
print('INFO: TFDV version -- {}'.format(tfdv.version.__version__))
print('INFO: Beam version -- {}'.format(pkg_resources.get_distribution("apache_beam").version))
print('INFO: Pyarrow version -- {}'.format(pkg_resources.get_distribution("pyarrow").version))
print('INFO: TFX-BSL version -- {}'.format(pkg_resources.get_distribution("tfx-bsl").version))

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  
    
def main(argv=None):

    # Setting Paths 

    # Set up some globals for gcs file
    HANDLER = 'gs://' # ../ for local data, gs:// for cloud data
        
    BASE_DIR = os.path.join(HANDLER, BUCKET, PIPELINE_VERSION)
    print('BASE_DIR')
    print(BASE_DIR)

    RUN_DIR = os.path.join(BASE_DIR, 'run', DATA_VERSION)

    STAGING_DIR = os.path.join(RUN_DIR, 'staging')
    OUTPUT_DIR = os.path.join(RUN_DIR, 'data_validation')

    FROZEN_STATS_PATH = os.path.join(BASE_DIR,'freeze', 'frozen_stats.txt')
    FROZEN_SCHEMA_PATH = os.path.join(BASE_DIR, 'freeze', 'frozen_schema.txt')
    DATA_STATS_PATH = os.path.join(OUTPUT_DIR, 'stats', 'data_stats.txt')
    DATA_SCHEMA_PATH = os.path.join(OUTPUT_DIR, 'schema', 'data_schema.txt')
    DATA_ANOMALIES_PATH = os.path.join(OUTPUT_DIR, 'anomalies', 'data_anomalies.txt')
    STATIC_HTML_PATH = os.path.join(OUTPUT_DIR, 'index.html')


    # Running on Google Cloud

    PATH_TO_WHL_FILE = [filename for filename in os.listdir('.') if filename.startswith('tensorflow_data_validation')]
    job_name = 'datavalidation-' + re.sub("_", "-", PIPELINE_VERSION) +     '-' + re.sub("_", "-", DATA_VERSION)

    # Create and set your PipelineOptions.
    options = PipelineOptions()

    # For Cloud execution, set the Cloud Platform project, job_name,
    # staging location, temp_location and specify DataflowRunner.
    google_cloud_options = options.view_as(GoogleCloudOptions)
    google_cloud_options.project = PROJECT
    google_cloud_options.job_name = job_name
    google_cloud_options.region = REGION
    google_cloud_options.staging_location = STAGING_DIR
    google_cloud_options.temp_location = STAGING_DIR
    options.view_as(WorkerOptions).subnetwork = 'regions/{}/subnetworks/default'.format(REGION)
    setup_options = options.view_as(SetupOptions)
    # PATH_TO_WHL_FILE should point to the downloaded tfdv wheel file.
    setup_options.extra_packages = PATH_TO_WHL_FILE
    options.view_as(StandardOptions).runner = RUNNER


    # 1- Computing descriptive data statistics

    stats_options = StatsOptions()
    stats_options.feature_whitelist = ["datetime","season","weather","daytype","temp",
                                    "atemp","humidity","windspeed","casual","registered",
                                    "count"]
                                    
    # Generating data statistics for initial dataset
    print('INFO: Generate & exporting data statistics to {}/'.format(DATA_STATS_PATH))
    data_stats =  tfdv.generate_statistics_from_csv(
        data_location=os.path.join(RAW_DATA_PATH, 'train.csv'), 
        output_path=DATA_STATS_PATH,
        pipeline_options=options,
        stats_options=stats_options)                                    


    # 2- Inferring a schema over the data

    ## 2.1- Inferring schema from data set

    data_schema = tfdv.infer_schema(data_stats)

    ## 2.2- Customizing schema

    ## 2.3- Schema Environments

    # casual, registered, count should be required during training, optional while serving

    # All features are by default in both TRAINING and SERVING environments.
    # Specify that 'partRootLabels' feature is not in SERVING environment.
    data_schema.default_environment.append('TRAINING')
    data_schema.default_environment.append('SERVING')
    tfdv.get_feature(data_schema, 'casual').not_in_environment.append('SERVING')
    tfdv.get_feature(data_schema, 'registered').not_in_environment.append('SERVING')
    tfdv.get_feature(data_schema, 'count').not_in_environment.append('SERVING')


    ## 2.4- Saving data schema

    def create_dir(path):
        '''
        A function that creates the directory of a provided path.
        (Might not be needed to save results to GS)
        '''
        path_dir = re.search('(.*)/', path).group(1)
        try:
            os.mkdir(path_dir)
        except OSError:
            print ("ERROR: Creation of the directory %s failed" % path_dir)
        else:
            print ("INFO: Successfully created the directory %s " % path_dir)


    if HANDLER != "gs://":
        create_dir(DATA_SCHEMA_PATH) # for local files only

    tfdv.write_schema_text(data_schema, DATA_SCHEMA_PATH)
    print('INFO: The data set schema was written to {}'.format(DATA_SCHEMA_PATH))


    ## 2.5- Loading frozen schema/stats

    # Check if frozen data schema exists otherwise create it from current data set
    try:
        frozen_schema = tfdv.load_schema_text(input_path=FROZEN_SCHEMA_PATH)
        print('INFO: Pipeline frozen data schema was loaded from {}'.format(FROZEN_SCHEMA_PATH))
    except:
        # First pipeline run, create new schema
        print('INFO: Frozen schema not found! First pipeline run! Saving current schema as frozen schema')
        frozen_schema = data_schema
        if HANDLER != "gs://":
            create_dir(FROZEN_SCHEMA_PATH)
        tfdv.write_schema_text(frozen_schema, FROZEN_SCHEMA_PATH)
        print('INFO: A new pipeline data schema was written to {}'.format(FROZEN_SCHEMA_PATH))

    # Check if frozen data statistics exist otherwise create them from current data set
    try:
        frozen_stats = tfdv.load_statistics(FROZEN_STATS_PATH)
        print('INFO: Pipeline frozen data statistics were loaded from {}'.format(FROZEN_STATS_PATH))
    except:
        # First pipeline run, create new schema
        print('INFO: No data statistics found at {}'.format(FROZEN_STATS_PATH))
        print('INFO: Frozen data statistics not found! First pipeline run! Saving current data statistics as frozen data statistics')
        frozen_stats=data_stats
        # Save new pipeline data stats
        tf.io.gfile.copy(
            DATA_STATS_PATH,
            FROZEN_STATS_PATH)
        print('INFO: New pipeline data statistics were written to {}/'.format(FROZEN_STATS_PATH))


    # 3- Checking the data for errors

    # Add a drift comparator to schema for catagorical features and set the threshold to 0.01
    tfdv.get_feature(frozen_schema, 'season').drift_comparator.infinity_norm.threshold = 0.01
    tfdv.get_feature(frozen_schema, 'weather').drift_comparator.infinity_norm.threshold = 0.01
    tfdv.get_feature(frozen_schema, 'daytype').drift_comparator.infinity_norm.threshold = 0.01


    # Detect schema anomalies and drift on new data set
    print('INFO: Check for schema anomalies and drift on new data set.')
    data_anomalies = tfdv.validate_statistics(
        statistics=data_stats,
        schema = frozen_schema,
        environment='TRAINING',
        previous_statistics=frozen_stats)

    if HANDLER != "gs://":
        create_dir(DATA_ANOMALIES_PATH) # for local use only
    tfdv.write_anomalies_text(data_anomalies, DATA_ANOMALIES_PATH)   
    print('INFO: Writing data anomalies to {}'.format(DATA_ANOMALIES_PATH))

    # 4- Saving results for Kubeflow Artifacts

    print('INFO: Rendering HTML artifacts.')
    features_html, domains_html = get_schema_html(data_schema)
    data_stats_drift_html = get_statistics_html(data_stats, frozen_stats, lhs_name="NEW_DATA", rhs_name="PREV_PREV")
    data_anomalies_html = get_anomalies_html(data_anomalies)


    # We can add some style to our html page.

    style="""
    <style>
    h1 {
        color:#0B6FA4;
    }
    h2 {
      color:#0B6FA4;
    }
    table.paleBlueRows {
        font-family: Arial, Helvetica, sans-serif;
        border: 1px solid #FFFFFF;
        text-align: left;
        border-collapse: collapse;
    }
    table.paleBlueRows td, table.paleBlueRows th {
        border: 1px solid #FFFFFF;
        padding: 3px 2px;
    }
    table.paleBlueRows tbody td {
        font-size: 13px;
    }
    table.paleBlueRows tr:nth-child(even) {
        background: #D0E4F5;
    }
    table.paleBlueRows thead {
        background: #0B6FA4;
        background: -moz-linear-gradient(top, #4893bb 0%, #237dad 66%, #0B6FA4 100%);
        background: -webkit-linear-gradient(top, #4893bb 0%, #237dad 66%, #0B6FA4 100%);
        background: linear-gradient(to bottom, #4893bb 0%, #237dad 66%, #0B6FA4 100%);
        border-bottom: 5px solid #FFFFFF;
    }
    table.paleBlueRows thead th {
        font-size: 15px;
        font-weight: bold;
        color: #FFFFFF;
        text-align: left;
        border-left: 2px solid #FFFFFF;
    }
    table.paleBlueRows thead th:first-child {
        border-left: none;
    }

    table.paleBlueRows tfoot td {
        font-size: 14px;
    }
    </style>
    """


    # Add the different html outputs to one html page:

    html = style +  '<h1>Schema</h1><h2>Features</h2>'  + features_html  + '<br><h2>Domains</h2>' + domains_html + '<br><h1>Dataset Statistics</h1>' +  data_stats_drift_html + '<br><h1>Dataset Anomalies</h1>' +  data_anomalies_html 


    # Write a HTML file to the component’s local filesystem and upload HTML file to GCS

    # Save and upload HTML file to GCS
    OUTPUT_FILE_PATH = './index.html'

    with open(OUTPUT_FILE_PATH, "wb") as f:
        f.write(html.encode('utf-8'))

    tf.io.gfile.copy(
        OUTPUT_FILE_PATH,
        STATIC_HTML_PATH,
        overwrite=True
    )


    # Our pipeline component must write a JSON file to the component’s local filesystem. We can do this at any point during the pipeline execution.

    metadata = {
    'outputs' : [{
      'type': 'web-app',
      'storage': 'gcs',
      'source': STATIC_HTML_PATH,
    }]
    }

    # Write output files for next steps in pipeline
    with file_io.FileIO('./mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)


    # Write data_version to txt output file to be used for next steps inputs in pipeline.

    with file_io.FileIO('./data_version.txt', 'w') as f:
        f.write(DATA_VERSION)

if __name__ == '__main__':

    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='Data Validation')
    parser.add_argument('--project', type=str, help='Name of the project to execute job.')
    parser.add_argument('--region', type=str, help='Name of the region to execute job in.')
    parser.add_argument('--raw_data_path', type=str, help='Path of source data.')
    parser.add_argument('--bucket', type=str, help='Pipeline metadata bucket where data are stored, and output written.') 
    parser.add_argument('--pipeline_version', type=str, help='Pipeline version.')
    parser.add_argument('--data_version', type=str, help='Data version.')
    parser.add_argument('--runner', type=str, default="DirectRunner", help='Type of runner.')

    args = parser.parse_args()

    # Input Arguments

    PROJECT = args.project
    REGION = args.region
    RAW_DATA_PATH = args.raw_data_path
    BUCKET = args.bucket
    PIPELINE_VERSION = args.pipeline_version
    DATA_VERSION = args.data_version
    RUNNER = args.runner # DirectRunner or DataflowRunner

    print('DATA_VERSION')
    print(DATA_VERSION)

    print('PIPELINE_VERSION')
    print(PIPELINE_VERSION)

    print('BUCKET')
    print(BUCKET)
    main()
    print('INFO: Done.')
    