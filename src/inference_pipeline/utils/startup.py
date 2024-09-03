import os
import boto3
import json
import tensorflow as tf

s3 = boto3.client("s3")

"""
function to load model and min-max-values needed for preprocessing and postprocessing from S3 bucket
"""
async def load_model(global_vars):
    s3_bucket = "blazemodelsregistry"
    s3_prefix = "optimiser/"
    local_model_path = "/tmp/model"

    if not os.path.exists(local_model_path):
        os.makedirs(local_model_path)

    highest_version = get_highest_version_folder(s3_bucket, s3_prefix)        

    model_s3_prefix = f"{s3_prefix}{highest_version}/model/"
    min_max_s3_key = f"{s3_prefix}{highest_version}/min_max_values.json"

    download_s3_directory(s3_bucket, model_s3_prefix, local_model_path)
    print(f"Model directory downloaded from S3: {model_s3_prefix}")

    min_max_path = os.path.join(local_model_path, "min_max_values.json")
    s3.download_file(s3_bucket, min_max_s3_key, min_max_path)

    with open(min_max_path, 'r') as f:
        global_vars['min_max_values'] = json.load(f)

    print(f"Loading model from {local_model_path}")
    model = tf.saved_model.load(local_model_path)
    global_vars['predict'] = model.signatures['serving_default']

    print(f"Model and min-max values loaded from version {highest_version}")

"""
helper function to download S3 directory to memory
"""
def download_s3_directory(bucket_name, s3_prefix, local_path):
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        for obj in page.get('Contents', []):
            s3_key = obj['Key']
            relative_path = os.path.relpath(s3_key, s3_prefix)
            local_file_path = os.path.join(local_path, relative_path)
            
            if s3_key.endswith('/'):
                os.makedirs(local_file_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                print(f"Downloading s3://{bucket_name}/{s3_key} to {local_file_path}")
                s3.download_file(bucket_name, s3_key, local_file_path)

"""
helper function to get the highest version folder in the S3 bucket
"""
def get_highest_version_folder(s3_bucket, s3_prefix):
    response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)
    version_numbers = []
    if 'Contents' in response:
        for obj in response['Contents']:
            folder_name = obj['Key'].split('/')[1]
            if folder_name.isdigit():
                version_numbers.append(int(folder_name))
    if version_numbers:
        return max(version_numbers)
    else:
        raise Exception("No version folders found in the S3 bucket")