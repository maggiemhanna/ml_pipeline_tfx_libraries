
from datetime import datetime
import os
import json
import shutil
import pkg_resources
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_model_analysis as tfma

print('INFO: TF version -- {}'.format(tf.__version__))
print('INFO: TFT version -- {}'.format(pkg_resources.get_distribution("tensorflow_transform").version))
print('INFO: TFMA version -- {}'.format(pkg_resources.get_distribution("tensorflow_model_analysis").version))
print('INFO: Beam version -- {}'.format(pkg_resources.get_distribution("apache_beam").version))
print('INFO: Pyarrow version -- {}'.format(pkg_resources.get_distribution("pyarrow").version))
print('INFO: tfx-bsl version -- {}'.format(pkg_resources.get_distribution("tfx-bsl").version))
print('INFO: absl-py version -- {}'.format(pkg_resources.get_distribution("absl-py").version))

# Features, labels, and key columns
NUMERIC_FEATURE_KEYS=["temp", "atemp", "humidity", "windspeed"] 
CATEGORICAL_FEATURE_KEYS=["season", "weather", "daytype"] 
KEY_COLUMN = "datetime"
LABEL_COLUMN = "count"

def transformed_name(key):
    return key 
    
def _get_session_config_from_env_var():
    """Returns a tf.ConfigProto instance that has appropriate device_filters
    set."""

    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))

    # Master should only communicate with itself and ps
    if (tf_config and 'task' in tf_config and 'type' in tf_config[
            'task'] and 'index' in tf_config['task']):
        if tf_config['task']['type'] == 'master':
            return tf.ConfigProto(device_filters=['/job:ps', '/job:master'])
        # Worker should only communicate with itself and ps
        elif tf_config['task']['type'] == 'worker':
            return tf.ConfigProto(device_filters=[
                '/job:ps',
                '/job:worker/task:%d' % tf_config['task']['index']
            ])
    return None


def get_dataset_size(file_path):
    """Function that fetchs the size of the Tfrecords dataset."""
    size = 1
    file_list = tf.io.gfile.glob(file_path)
    for file in file_list:
        for record in tf.compat.v1.io.tf_record_iterator(file, options=tf.io.TFRecordOptions(
    compression_type='GZIP')):
            size += 1
    return size


# Train and Evaluate input functions
    
# Train and Evaluate input functions
    
def input_fn(data_path, label_column, tf_transform_output, batch_size, mode = tf.estimator.ModeKeys.TRAIN):
    """Create an input function reading TFRecord files using the data API.
    Args:
        data_path: path of the data in tfrecords format
        mode: tf estimator mode key
        batch_size: number of observations in batch

    Returns:
        input_fn: data input function
    """
    
    features_spec = tf_transform_output.transformed_feature_spec()

    def _input_fn():
        # Create list of files in the data path
        file_list = tf.io.gfile.glob(data_path)

        # Create dataset from file list
        dataset = tf.data.TFRecordDataset(filenames=file_list, compression_type = "GZIP", num_parallel_reads=5)
        def parse_example(example):
            parsed_features = tf.io.parse_single_example(example, features_spec)
            label = parsed_features.pop(label_column)
            return parsed_features, label
          
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely, we'll set this in train spec
            dataset = dataset.shuffle(buffer_size=10*batch_size)
        else:
            num_epochs = 1 # end-of-input after one epoch

        dataset = dataset.repeat(num_epochs)
        dataset = dataset.map(parse_example, num_parallel_calls=5)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=1)

    
        return dataset
    
    return _input_fn


# Feature Engineering

def create_feature_columns(tf_transform_output):
    
    numeric_columns = [
      tf.feature_column.numeric_column(transformed_name(key))
      for key in NUMERIC_FEATURE_KEYS
    ]
    
    categorical_columns = [
      tf.feature_column.categorical_column_with_vocabulary_file(
        transformed_name(key), 
        vocabulary_file=tf_transform_output.vocabulary_file_by_name(
            vocab_filename=key), 
        dtype=tf.dtypes.string,
        default_value=None, 
        num_oov_buckets=0)
      for key in CATEGORICAL_FEATURE_KEYS
    ]
    
    indicator_columns = [
      tf.feature_column.indicator_column(categorical_column)
      for categorical_column in categorical_columns
    ]
       
    feature_columns = numeric_columns + indicator_columns

    return feature_columns


def rmse(labels, predictions):
    pred_values = predictions['predictions']
    rmse = tf.keras.metrics.RootMeanSquaredError(name="rmse")
    rmse.update_state(y_true=labels, y_pred=pred_values)
    return {'rmse': rmse}

def rmse_2(labels, predictions):
    pred_values = predictions['predictions']
    rmse = tf.compat.v1.metrics.root_mean_squared_error(labels, pred_values)
    return {'rmse': rmse}


def mae(labels, predictions):
    pred_values = tf.squeeze(input = predictions["predictions"], axis = -1)
    mae = tf.keras.metrics.MeanAbsoluteError(name="mae")
    mae.update_state(y_true=labels, y_pred=pred_values)
    return {'mae': mae}


def create_estimator_model(output_dir, feature_columns, hidden_units, run_config):
    model = tf.estimator.DNNRegressor(
        model_dir = output_dir,
        feature_columns = feature_columns,
        hidden_units = hidden_units, # specify neural architecture
        config = run_config
    )
    
    return model

def serving_input_fn(tf_transform_output, label_column):
    """Creates an input function reading from raw data.

    Args:
    tf_transform_output: Wrapper around output of tf.Transform.

    Returns:
    The serving input function.
    """
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    # Remove label since it is not available during serving.
    raw_feature_spec.pop(label_column)

    def _input_fn():
        """Input function for serving."""
        # Get raw features by generating the basic serving input_fn and calling it.
        # Here we generate an input_fn that expects a parsed Example proto to be fed
        # to the model at serving time.  See also
        # tf.estimator.export.build_raw_serving_input_receiver_fn.
        raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
            raw_feature_spec, default_batch_size=None)
        serving_input_receiver = raw_input_fn()

        # Apply the transform function that was used to generate the materialized
        # data.
        raw_features = serving_input_receiver.features
        transformed_features = tf_transform_output.transform_raw_features(
            raw_features)

        return tf.estimator.export.ServingInputReceiver(
            transformed_features, serving_input_receiver.receiver_tensors)

    return _input_fn

def eval_input_receiver_fn(tf_transform_output, label_column):
    """Function that defines an input placeholder,
     parses and returns features and labels for evaluation."""
    
    def _input_fn():

        serialized_tf_example = tf.compat.v1.placeholder(
            dtype=tf.string, shape=[None], name='input_example_placeholder')

        # This *must* be a dictionary containing a single key 'examples', which
        # points to the input placeholder.
        receiver_tensors = {'examples': serialized_tf_example}

        transformed_feature_spec = tf_transform_output.transformed_feature_spec()
             
        features = tf.io.parse_example(serialized_tf_example, transformed_feature_spec)

        return tfma.export.EvalInputReceiver(
            features=features,
            receiver_tensors=receiver_tensors,
            labels=features[label_column])

    return _input_fn

# train and evaluate

def train_and_evaluate(params):
    
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO) # so loss is printed during training

    # Extract params from task.py
    DATA_DIR = params["data_dir"]
    OUTPUT_DIR = params["output_dir"]
    HIDDEN_UNITS_1 = params["hidden_units_1"]
    HIDDEN_UNITS_2 = params["hidden_units_2"]
    HIDDEN_UNITS_3 = params["hidden_units_3"]
    BATCH_SIZE = params["batch_size"]
    NUM_EPOCHS = params["num_epochs"]
    LEARNING_RATE = params["learning_rate"]

    # Setting up paths 
    TRAIN_PATH = DATA_DIR+'/train*'
    VAL_PATH = DATA_DIR+'/val*'
    TEST_PATH = DATA_DIR+'/test*'

    # Define key and label columns
    KEY_COLUMN = 'datetime'
    LABEL_COLUMN = 'count'
    
    # Training set size
    TRAIN_SIZE = get_dataset_size(TRAIN_PATH)

    NUM_STEPS = TRAIN_SIZE / BATCH_SIZE * NUM_EPOCHS # total steps for which to train model
    CHECKPOINTS_STEPS = 16 # checkpoint every N steps
    
    tf_transform_output = tft.TFTransformOutput(os.path.join(DATA_DIR, 'tft_output'))

    FEATURE_COLUMNS = create_feature_columns(tf_transform_output)
    
    run_config = tf.estimator.RunConfig(
        tf_random_seed = 1, # for reproducibility
        save_checkpoints_steps = CHECKPOINTS_STEPS, # checkpoint every N steps
        save_summary_steps = int(CHECKPOINTS_STEPS/5),
        session_config = _get_session_config_from_env_var()
        )
    
    estimator = create_estimator_model(OUTPUT_DIR, FEATURE_COLUMNS, [HIDDEN_UNITS_1, 
                                       HIDDEN_UNITS_2, HIDDEN_UNITS_3], run_config)
    
    estimator = tf.estimator.add_metrics(estimator = estimator, metric_fn = rmse) 
    estimator = tf.estimator.add_metrics(estimator = estimator, metric_fn = mae) 

    train_input_fn = input_fn(TRAIN_PATH, LABEL_COLUMN, tf_transform_output, 
                              BATCH_SIZE, tf.estimator.ModeKeys.TRAIN)
    val_input_fn = input_fn(VAL_PATH, LABEL_COLUMN, tf_transform_output, 
                            BATCH_SIZE, tf.estimator.ModeKeys.EVAL)
    
    train_spec = tf.estimator.TrainSpec(
        input_fn = train_input_fn,
        max_steps = NUM_STEPS)

    exporter = tf.estimator.LatestExporter(name = 'exporter', 
               serving_input_receiver_fn = serving_input_fn(tf_transform_output, LABEL_COLUMN))

    eval_spec = tf.estimator.EvalSpec(
        input_fn = val_input_fn,
        steps = CHECKPOINTS_STEPS, # Number of steps to run evalution for at each checkpoint
        start_delay_secs = 1, # wait at least N seconds before first evaluation (default 120)
        throttle_secs = 16, # wait at least N seconds before each subsequent evaluation (default 600)
        exporters = exporter) # export SavedModel once at the end of training
    
    tf.estimator.train_and_evaluate(
        estimator = estimator, 
        train_spec = train_spec, 
        eval_spec = eval_spec) 
    
    # Also export the EvalSavedModel
    tfma.export.export_eval_savedmodel(
        estimator=estimator, export_dir_base=OUTPUT_DIR + '/eval_saved_model/',
        eval_input_receiver_fn=eval_input_receiver_fn(tf_transform_output, LABEL_COLUMN))
    
