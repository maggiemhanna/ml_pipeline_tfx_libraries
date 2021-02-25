#!/usr/bin/env python3

# Copyright 2020 RENAULT DIGITAL. All Rights Reserved.
# =============================================================================

"""Utilitary functions for model analysis."""

import gcsfs
import re
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from google.cloud import storage, bigquery
#from google.cloud import bigquery_storage_v1beta1

from trainer import model
from trainer.model import Params, model_fn





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
    blobs = storage_client.list_blobs(bucket_name, prefix=source_dir)

    # Copy all blobs from source_dir to destination_dir
    for blob in blobs:

        # Define path inside destination directory
        short_destination = blob.name.replace(source_dir, '')

        # Copy blob
        blob_copy = storage_bucket.copy_blob(
            blob, storage_bucket, destination_dir+'/'+short_destination)

        if not mute:
            print("INFO: Blob {} in bucket {} copied to blob {} in bucket {}.\n".format(
                blob.name, bucket_name, blob_copy.name, bucket_name))


def delete_gcs_dir(bucket_name, dir_path, mute=True):
    """Deletes a GCS directory with all its blobs.
    
    Args:
        bucket_name (string): bucket name
        dir_path: path to directory to delete (without gs://{bucket_name}/)
    """

    # Get full gcs uri of directory to delete
    dir_uri = 'gs://'+bucket_name+'/'+dir_path

    # Delete directory if exists
    if tf.io.gfile.exists(dir_uri):

        # Create storage client and set bucket
        storage_client = storage.Client()

        # List all blobs in
        blobs = storage_client.list_blobs(bucket_name, prefix=dir_path)

        # Delete all blobs with directory include itself
        for blob in blobs:
            blob.delete()

        if not mute:
            print('INFO: The following GCS directory was entirely deleted {}'.format(
                dir_uri))
    else:
        raise ValueError('GCS directory {} does not exist.'.format(dir_uri))

