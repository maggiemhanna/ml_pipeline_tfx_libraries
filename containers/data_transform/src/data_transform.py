#!/usr/bin/env python3

# Importing Librarires

import argparse
from datetime import datetime

import pkg_resources
import tempfile
import pprint
import os
import json

import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
import tensorflow_data_validation as tfdv
import apache_beam as beam


from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.tf_metadata import metadata_io
from apache_beam.io import tfrecordio
from tensorflow.python.lib.io import file_io

print('INFO: TF version -- {}'.format(tf.__version__))
print('INFO: TFT version -- {}'.format(tft.version.__version__))
print('INFO: TFDV version -- {}'.format(tfdv.version.__version__))
print('INFO: Apache Beam version -- {}'.format(beam.version.__version__))
print('INFO: Pyarrow version -- {}'.format(pkg_resources.get_distribution("pyarrow").version))
print('INFO: TFX-BSL version -- {}'.format(pkg_resources.get_distribution("tfx-bsl").version))

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
def main(argv=None):

    # Setting Paths 

    # Set up some globals for gcs file
    HANDLER = 'gs://' # ../ for local data, gs:// for cloud data

    RAW_TRAIN_PATH = os.path.join(RAW_DATA_PATH, "train.csv")
    RAW_VAL_PATH = os.path.join(RAW_DATA_PATH, "val.csv")
    RAW_TEST_PATH = os.path.join(RAW_DATA_PATH, "test.csv")

    BASE_DIR = os.path.join(HANDLER, BUCKET, PIPELINE_VERSION)
    RUN_DIR = os.path.join(BASE_DIR, 'run', DATA_VERSION)

    STAGING_DIR = os.path.join(RUN_DIR, 'staging')
    OUTPUT_DIR = os.path.join(RUN_DIR, 'data_transform')

    RAW_SCHEMA_PATH = RUN_DIR+'/data_validation/schema/data_schema.txt'

    # Features, labels, and key columns
    NUMERIC_FEATURE_KEYS=["temp", "atemp", "humidity", "windspeed"] 
    CATEGORICAL_FEATURE_KEYS=["season", "weather", "daytype"] 
    KEY_COLUMN = "datetime"
    LABEL_COLUMN = "count"

    def transformed_name(key):
        return key 

    # Transformation Functions

    # A function that allows to split dataset
    def split_dataset(row, num_partitions, ratio):
        assert num_partitions == len(ratio)

        bucket = hash(row['datetime'][0]) % sum(ratio)
        total = 0
        for i, part in enumerate(ratio):
            total += part
            if bucket < total:
                return i
        return len(ratio) - 1

    # A function to scale numerical features and label encode categorical features
    def preprocessing_fn(inputs):
        
        outputs = {}
        
        for key in NUMERIC_FEATURE_KEYS:
            outputs[transformed_name(key)] = tft.scale_to_z_score(squeeze(inputs[key]))
        
        for key in CATEGORICAL_FEATURE_KEYS:    
            outputs[transformed_name(key)] = squeeze(inputs[key])
            tft.vocabulary(inputs[key], vocab_filename=key)    

        outputs[transformed_name(LABEL_COLUMN)] = squeeze(inputs[LABEL_COLUMN])
        outputs[transformed_name(KEY_COLUMN)] = squeeze(inputs[KEY_COLUMN])

        return outputs

    def squeeze(x):
        return tf.squeeze(x, axis=1)

    # Create ML dataset using tf.transform and Dataflow

    job_name = 'bike-sharing-data-transform' + '-' + datetime.now().strftime('%y%m%d-%H%M%S')    
    in_test_mode = True

    if RUNNER == 'DirectRunner':
        import shutil
        print('Launching local job ... hang on')
        #OUTPUT_DIR = './preproc_tft'
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    if RUNNER == 'DataflowRunner':
        print('Launching Dataflow job {} ... hang on'.format(job_name))
    # OUTPUT_DIR = 'gs://{0}/taxifare/preproc_tft/'.format(BUCKET)
        import subprocess
        subprocess.call('gsutil rm -r {}'.format(OUTPUT_DIR).split())

    options = {
        'staging_location': os.path.join(OUTPUT_DIR, 'tmp', 'staging'),
        'temp_location': os.path.join(OUTPUT_DIR, 'tmp'),
        'job_name': job_name,
        'project': PROJECT,
        'region': REGION,
        'max_num_workers': 4,
        'teardown_policy': 'TEARDOWN_ALWAYS',
        'no_save_main_session': True,
        'requirements_file': 'requirements.txt'
    }
    opts = beam.pipeline.PipelineOptions(flags=[], **options)

    # Load raw data schema and convert to tft metadata
    raw_schema = tfdv.load_schema_text(input_path=RAW_SCHEMA_PATH)
    raw_metadata = dataset_metadata.DatasetMetadata(raw_schema)
    ordered_columns = [i.name for i in raw_schema.feature]

    converter = tft.coders.CsvCoder(ordered_columns, raw_schema)

    with beam.Pipeline(RUNNER, options=opts) as p:
        with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
            
            # Read raw train data from csv 
            raw_train_data = (p 
            | 'ReadTrainData' >> beam.io.ReadFromText(RAW_TRAIN_PATH, skip_header_lines=1)
            | 'DecodeTrainData' >> beam.Map(converter.decode))
                            
            # avoid data leakage from raw_metadata! split tests before data validation
            raw_train_dataset = (raw_train_data, raw_metadata)
            
            # Analyze and transform data 
            transformed_train_dataset, transform_fn = (  
                raw_train_dataset | "TransformTrainData" >> tft_beam.AnalyzeAndTransformDataset(
                    preprocessing_fn)) 
            
            transformed_train_data, transformed_metadata = transformed_train_dataset
        
            # Save transformed train data to disk in efficient tfrecord format
            transformed_train_data | 'WriteTrainData' >> tfrecordio.WriteToTFRecord(
                os.path.join(OUTPUT_DIR, 'train'), file_name_suffix='.gz',
                coder=example_proto_coder.ExampleProtoCoder(
                    transformed_metadata.schema))
        
            # save transformation function to disk for use at serving time
            transform_fn | 'WriteTransformFn' >> tft_beam.WriteTransformFn(
                os.path.join(OUTPUT_DIR, 'tft_output'))                
            raw_metadata | 'WriteDataMetadata' >> tft_beam.WriteMetadata(
                os.path.join(OUTPUT_DIR, 'tft_output', 'metadata'), pipeline=p)  
            
            raw_val_data = (p 
            | 'ReadValData' >> beam.io.ReadFromText(RAW_VAL_PATH, skip_header_lines=1)
            | 'DecodeValData' >> beam.Map(converter.decode))
            
            # avoid data leakage from raw_metadata! split tests before data validation        
            raw_val_dataset = (raw_val_data, raw_metadata)

            # Transform val data
            transformed_val_dataset = (
                (raw_val_dataset, transform_fn) | "TransformValData" >> tft_beam.TransformDataset()
                )
            
            transformed_val_data, _ = transformed_val_dataset

            # Save transformed train data to disk in efficient tfrecord format
            transformed_val_data | 'WriteValData' >> tfrecordio.WriteToTFRecord(
                os.path.join(OUTPUT_DIR, 'val'), file_name_suffix='.gz',
                coder=example_proto_coder.ExampleProtoCoder(
                    transformed_metadata.schema))
            
            raw_test_data = (p 
            | 'ReadTestData' >> beam.io.ReadFromText(RAW_TEST_PATH, skip_header_lines=1)
            | 'DecodeTestData' >> beam.Map(converter.decode))
            
            # avoid data leakage from raw_metadata! split tests before data validation        
            raw_test_dataset = (raw_test_data, raw_metadata)
            
            # Transform test data
            transformed_test_dataset = (
                (raw_test_dataset, transform_fn) | "TransformTestData" >> tft_beam.TransformDataset()
                )
            
            transformed_test_data, _ = transformed_test_dataset
            
            # Save transformed train data to disk in efficient tfrecord format
            transformed_test_data | 'WriteTestData' >> tfrecordio.WriteToTFRecord(
                os.path.join(OUTPUT_DIR, 'test'), file_name_suffix='.gz',
                coder=example_proto_coder.ExampleProtoCoder(
                    transformed_metadata.schema))                           


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

    main()
    print('INFO: Done.')
    