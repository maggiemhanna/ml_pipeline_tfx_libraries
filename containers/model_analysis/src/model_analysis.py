#!/usr/bin/env python3

# Importing Librarires

import argparse
import json
import os
import logging
import re
import pandas as pd
import tensorflow as tf
import tensorflow_model_analysis as tfma
import apache_beam as beam

from tensorflow.python.lib.io import file_io
from ipywidgets.embed import embed_data
from io import BytesIO

import pkg_resources
from google.cloud import storage

print('TF version: {}'.format(tf.__version__))
print('TFMA version: {}'.format(pkg_resources.get_distribution("tensorflow_model_analysis").version))
print('INFO: Beam version -- {}'.format(pkg_resources.get_distribution("apache_beam").version))
print('INFO: Pyarrow version -- {}'.format(pkg_resources.get_distribution("pyarrow").version))

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  
    
def main(argv=None):

    # Setting Paths 

    # Set up some globals for gcs file
    HANDLER = 'gs://' # ../ for local data, gs:// for cloud data

    BASE_DIR = os.path.join(HANDLER, BUCKET, PIPELINE_VERSION)
    RUN_DIR = os.path.join(BASE_DIR, 'run', DATA_VERSION)
    DATA_DIR = os.path.join(RUN_DIR, 'data_transform')
    MODEL_DIR = os.path.join(RUN_DIR, 'model_training', MODEL_VERSION, str(TRIAL_ID) if TRIAL_ID is not None else "")
    OUTPUT_DIR = os.path.join(RUN_DIR, 'model_analysis', MODEL_VERSION)

    TEST_PATH = DATA_DIR+'/test*'

    # Features, labels, and key columns
    NUMERIC_FEATURE_KEYS=["temp", "atemp", "humidity", "windspeed"] 
    CATEGORICAL_FEATURE_KEYS=["season", "weather", "daytype"] 
    KEY_COLUMN = "datetime"
    LABEL_COLUMN = "count"

    # 2- Evaluation slices

    ## 2.1- Specify model to use for evaluation

    # Specify model to use for evaluation
    eval_saved_model_path = MODEL_DIR+'/eval_saved_model/'
    eval_saved_model_path = eval_saved_model_path + list_dirs(eval_saved_model_path, '(\d)+')[-1]
    eval_shared_model = tfma.default_eval_shared_model(eval_saved_model_path)

    # copy eval_saved_model to output_dir
    copy_gcs_dir(BUCKET, eval_saved_model_path+'/', OUTPUT_DIR+'/eval_saved_model', mute=False)


    ## 2.2- Defining slices of evaluation

    feature_slices = [["season"], ["weather"], ['daytype']]

    # Defining slices of evaluation
    # An empty spec is required for the 'Overall' slice 
    slices = [tfma.slicer.SingleSliceSpec()] + [tfma.slicer.SingleSliceSpec(columns=x) for x in feature_slices]


    ## 2.3- Write tfma evaluation results

    ### Method 1: using run_model_analysis

    #eval_dir = OUTPUT_DIR+ \
    #    '/eval_result'

    #eval_result = tfma.run_model_analysis(
    #    eval_shared_model=eval_shared_model,
    #    data_location=TEST_PATH,
    #    file_format='tfrecords',
    #    slice_spec=slices,
    #    output_path=eval_dir
    #)


    ### Method 2: Apache Beam pipeline

    eval_dir = OUTPUT_DIR + '/eval_result'

    with beam.Pipeline(runner='DirectRunner') as p:
        _ = (p
            # You can change the source as appropriate, e.g. read from BigQuery.
            | 'ReadData' >> beam.io.ReadFromTFRecord(TEST_PATH)
            | 'ExtractEvaluateAndWriteResults' >>
            tfma.ExtractEvaluateAndWriteResults(
                eval_shared_model=eval_shared_model,
                output_path=eval_dir,
                compute_confidence_intervals=False,
                slice_spec=slices
            ))

    print("INFO: TFMA Evaluation Job exported to {}".format(eval_dir))

    # Load slices evaluation results
    eval_result = tfma.load_eval_result(eval_dir)
    print("INFO: TFMA Evaluation Results loaded from {}".format(eval_result))


    # 3- Tracking Model Performance Over Time

    print("INFO: Comparing with previous trial results.")

    # Get list of all runs within the same pipeline version
    eval_dirs = []
    runs = list_dirs(BASE_DIR+'/run', '(\d){6}_(\d){6}')

    # Get list of all evaluation results (only best hypertune trials)
    for run in runs:
        run_eval_dir = BASE_DIR+'/run/'+run+'/model_analysis/'
        models = list_dirs(run_eval_dir, '(\d){6}_(\d){6}')
        for model in models: 
            model_eval_dir = run_eval_dir + model + '/eval_result'
            if tf.io.gfile.exists(model_eval_dir + '/eval_config.json'):
                eval_dirs = eval_dirs + [model_eval_dir]

    # Load best trial evaluation results
    eval_results = tfma.load_eval_results(
        eval_dirs,
        tfma.constants.MODEL_CENTRIC_MODE
    )

    print("INFO: Measuring performance on new test data.")

    # 4- Tracking Model Performance Over Today data

    def get_eval_result(eval_saved_model_dir, output_dir, data_dir, slice_spec):
        """Runs tfma model analysis locally to compute slicing metrics."""

        eval_shared_model = tfma.default_eval_shared_model(eval_saved_model_path=eval_saved_model_dir)

        return tfma.run_model_analysis(
            eval_shared_model=eval_shared_model,
            data_location=data_dir,
            file_format='tfrecords',
            slice_spec=slice_spec,
            output_path=output_dir,
            extractors=None)

    print("INFO: Comparing with previous trial results.")

    eval_today_dirs = []
    eval_today_results_dict = {}

    # Get list of all runs within the same pipeline version
    runs = list_dirs(BASE_DIR+'/run', '(\d){6}_(\d){6}')

    # Get list of all evaluation results (only best hypertune trials)
    for run in runs:
        
        run_eval_saved_model_dir = BASE_DIR+'/run/'+run+'/model_analysis/'
        models = list_dirs(run_eval_saved_model_dir, '(\d){6}_(\d){6}')
        for model in models: 
            model_eval_saved_model_dir = run_eval_saved_model_dir + model + '/eval_saved_model'
            if tf.io.gfile.exists(model_eval_saved_model_dir + '/saved_model.pb'):
                
                run_eval_today_dir = OUTPUT_DIR + '/eval_today_result/eval_result_{}_{}'.format(run, model)
                eval_today_results_dict[run, model] = get_eval_result(model_eval_saved_model_dir, run_eval_today_dir, TEST_PATH, slices)               
                eval_today_dirs = eval_today_dirs + [run_eval_today_dir]
        
        
    # Load evaluation results on new test data
    eval_today_results = tfma.load_eval_results(
        eval_today_dirs,
        tfma.constants.MODEL_CENTRIC_MODE
    )

    # 5- Write Evaluation Results to Artifacts
    generate_static_html_output(eval_result, eval_results, eval_today_results,
        slices, OUTPUT_DIR)
    print("INFO: HTML artifacts generated successfully.")


    # OK or KO Deploy Flag

    dfPerf = pd.DataFrame(columns=['pipeline_version', 'data_version', 'model_version', 'rmse'])

    for version in list(eval_today_results_dict.keys()):
        dfPerf = dfPerf.append({'pipeline_version': PIPELINE_VERSION, 
                            'data_version': version[0], 
                            'model_version': version[1], 
                            'eval_data_version': DATA_VERSION,
                            'rmse': eval_today_results_dict[version][0][0][1]['']['']['rmse']['doubleValue']}, ignore_index=True)

    if dfPerf.loc[dfPerf.model_version == MODEL_VERSION, "rmse"].item() == min(dfPerf["rmse"]):
        deployment_flag = 'OK'
    else: 
        deployment_flag = 'KO'


    dfPerf.to_csv("dfPerf.csv", index=False)

    dfPerf_dir = OUTPUT_DIR + '/perf_models_today.csv'

    # upload file to gs
    tf.io.gfile.copy(
        "dfPerf.csv",
        dfPerf_dir,
        overwrite=True
    )   
        
    # write table 

    dfPerf_schema = dfPerf.columns.to_list()
    static_html_path = OUTPUT_DIR+'/'+_OUTPUT_HTML_FILE

    metadata = {
        'outputs' : [
            {
                'type': 'web-app',
                'storage': 'gcs',
                'source': static_html_path,
            },
            {
                'type': 'table',
                'storage': 'gcs',
                'format': 'csv',
                'header': dfPerf_schema,
                'source': dfPerf_dir
            }
        ]
    }
            
    with file_io.FileIO('mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)

    with file_io.FileIO('./data_version.txt', 'w') as f:
        f.write(DATA_VERSION)

    with file_io.FileIO('./model_version.txt', 'w') as f:
        f.write(MODEL_VERSION)

    with file_io.FileIO('./trial_id.txt', 'w') as f:
        f.write(TRIAL_ID)

    with file_io.FileIO('./deployment_flag.txt', 'w') as f:
        f.write(deployment_flag)

# Global Variables & Functions

_OUTPUT_HTML_FILE = "index.html"

# html scripts for analysis artifacts
_STATIC_HTML_TEMPLATE = """
<html>
<head>
    <title>Slicing Metrics</title>
    <!-- Load RequireJS, used by the IPywidgets for dependency management -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"
            integrity="sha256-Ae2Vz/4ePdIu6ZyI/5ZGsYnb+m0JlOmKPjt6XZ9JJkA="
            crossorigin="anonymous">
    </script>
    <!-- Load IPywidgets bundle for embedding. -->
    <script src="https://unpkg.com/@jupyter-widgets/html-manager@^*/dist/embed-amd.js"
            crossorigin="anonymous">
    </script>
    
    <!-- Load vulcanized tfma code from prebuilt js file. -->
    <script src="https://raw.githubusercontent.com/tensorflow/model-analysis/v0.22.0/tensorflow_model_analysis/static/vulcanized_tfma.js">
    </script> 
    <!-- Load IPywidgets bundle for embedding. -->
    <script>
    require.config({{
        paths: {{
        "tfma_widget_js": "https://raw.githubusercontent.com/tensorflow/model-analysis/v0.22.0/tensorflow_model_analysis/static/index", 
        }} 
    }});
    </script>
    <!-- The state of all the widget models on the page -->
    <script type="application/vnd.jupyter.widget-state+json">
    {manager_state}
    </script>
</head>
<body>
    <h1>Slicing Metrics</h1>
    {slicing_widget_views}
    <h1>Comparison with Past Trial Results</h1>
    {ts_widget_views}
    <h1>How Does It Look Today</h1>
    {ts_today_widget_views}
</body>
</html>
"""

_SLICING_METRICS_WIDGET_TEMPLATE = """
    <div id="slicing-metrics-widget-{0}">
    <script type="application/vnd.jupyter.widget-view+json">
        {1}
    </script>
    </div>
"""

_TS_METRICS_WIDGET_TEMPLATE = """
    <div id="ts-metrics-widget-{0}">
    <script type="application/vnd.jupyter.widget-view+json">
        {1}
    </script>
    </div>
"""

_TS_TODAY_METRICS_WIDGET_TEMPLATE = """
    <div id="ts-today-metrics-widget-{0}">
    <script type="application/vnd.jupyter.widget-view+json">
        {1}
    </script>
    </div>
"""

def list_dirs(path, pattern):
    """Function that returns all files in GCS directory corresponding to some pattern."""

    runs = tf.io.gfile.listdir(path)
    runs = [re.sub('/', '', run) for run in runs]
    runs = [re.match(pattern, run).group(0)
            for run in runs if re.match(pattern, run) != None]
    runs.sort()

    return runs

def copy_gcs_dir(bucket_name, source_dir, destination_dir, mute=False):
    """Copies a GCS directory with all its blobs.
    
    Args:
        bucket_name (string): bucket name
        source_dir: path to directory containing the blobs to copy (without gs://{bucket_name}/)
        destination_dir: path directory where to copy blobs (without gs://{bucket_name}/)
    """

    # Create storage client and set bucket
    storage_client = storage.Client()
    storage_bucket = storage_client.bucket(bucket_name)

    # List all blobs in source directory
    generic_source_dir = source_dir.replace('gs://'+bucket_name+'/', '')
    generic_destination_dir = destination_dir.replace('gs://'+bucket_name+'/', '')

    blobs = storage_client.list_blobs(bucket_name, prefix=generic_source_dir)

    # Copy all blobs from source_dir to destination_dir
    for blob in blobs:

        # Define path inside destination directory
        short_destination = blob.name.replace(generic_source_dir, '')

        # Copy blob
        blob_copy = storage_bucket.copy_blob(
            blob, storage_bucket, generic_destination_dir+'/'+short_destination)

        if not mute:
            print("INFO: Blob {} in bucket {} copied to blob {} in bucket {}.\n".format(
                blob.name, bucket_name, blob_copy.name, bucket_name))
                    

def generate_static_html_output(eval_result, eval_results, eval_today_results,
     slicing_specs, html_output_dir):
    """W R I T E  D O C S T R I N G!!"""

    # Slicing Metrics Eval Result
    if slicing_specs is not None:
        slicing_metrics_views = [
          tfma.view.render_slicing_metrics(
              eval_result,
              slicing_spec=slicing_spec)
          for slicing_spec in slicing_specs
        ]
    else:
        slicing_metrics_views = [
          tfma.view.render_slicing_metrics(eval_result)
        ]
        
    # Time series Eval Result
    ts_metrics_view = tfma.view.render_time_series(
            eval_results, display_full_path=False)
    ts_today_metrics_view = tfma.view.render_time_series(
            eval_today_results, display_full_path=False)
        
    slicing_data = embed_data(views=slicing_metrics_views)
    manager_state = json.dumps(slicing_data['manager_state'])
    slicing_widget_views = [json.dumps(view) for view in slicing_data['view_specs']]
    slicing_views_html = ""
    
    for idx, view in enumerate(slicing_widget_views):
        slicing_views_html += _SLICING_METRICS_WIDGET_TEMPLATE.format(idx, view)
    
    ts_data = embed_data(views=ts_metrics_view)
    ts_today_data = embed_data(views=ts_today_metrics_view)
    ts_widget_views = [json.dumps(view) for view in ts_data['view_specs']]
    ts_today_widget_views = [json.dumps(view) for view in ts_today_data['view_specs']]
    ts_views_html = ""
    ts_today_views_html = ""

    for idx, view in enumerate(ts_widget_views):
        ts_views_html += _TS_METRICS_WIDGET_TEMPLATE.format(idx, view)
    for idx, view in enumerate(ts_today_widget_views):
        ts_today_views_html += _TS_TODAY_METRICS_WIDGET_TEMPLATE.format(idx, view)       
    
    rendered_template = _STATIC_HTML_TEMPLATE.format(
        manager_state=manager_state,
        slicing_widget_views=slicing_views_html,
        ts_widget_views=ts_views_html,
        ts_today_widget_views=ts_today_views_html)
    
    static_html_path = html_output_dir+'/'+_OUTPUT_HTML_FILE
       
    with open(os.path.join(".", _OUTPUT_HTML_FILE), "wb") as f:
        f.write(rendered_template.encode('utf-8'))

    print("INFO: Writing html artifacts to {}".format(static_html_path))    
    # upload file to gs
    tf.io.gfile.copy(
        _OUTPUT_HTML_FILE,
        static_html_path,
        overwrite=True
    )   


if __name__ == '__main__':

    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='Data Validation')
    parser.add_argument('--project', type=str, help='Name of the project to execute job.')
    parser.add_argument('--region', type=str, help='Name of the region to execute job in.')
    parser.add_argument('--bucket', type=str, help='Pipeline metadata bucket where data are stored, and output written.') 
    parser.add_argument('--pipeline_version', type=str, help='Pipeline version.')
    parser.add_argument('--data_version', type=str, help='Data version.')
    parser.add_argument('--model_version', type=str, help='Model version.')
    parser.add_argument('--trial_id', type=str, help='Best Trial Id.')
    parser.add_argument('--runner', type=str, default="DirectRunner", help='Type of runner.')

    args = parser.parse_args()

    # Input Arguments

    PROJECT = args.project
    REGION = args.region
    BUCKET = args.bucket
    PIPELINE_VERSION = args.pipeline_version
    DATA_VERSION = args.data_version
    MODEL_VERSION = args.model_version
    TRIAL_ID = args.trial_id
    RUNNER = args.runner # DirectRunner or DataflowRunner

    main()
    print('INFO: Done.')
    