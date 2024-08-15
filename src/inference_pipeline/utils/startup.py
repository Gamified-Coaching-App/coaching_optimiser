import os
import boto3
import tarfile
import json
import tensorflow as tf

# Initialize the S3 client
s3 = boto3.client("s3")

async def load_model(global_vars):
    s3_bucket = "blazemodelsregistry"
    s3_prefix = "optimiser/"
    local_model_path = "/tmp/model"

    # Create the directory if it doesn't exist
    if not os.path.exists(local_model_path):
        os.makedirs(local_model_path)

    highest_version = get_highest_version_folder(s3_bucket, s3_prefix)        

    # Construct S3 keys for the latest version
    model_s3_key = f"{s3_prefix}{highest_version}/model.tar.gz"
    min_max_s3_key = f"{s3_prefix}{highest_version}/min_max_values.json"

    # Download the model.tar.gz from S3
    model_tar_path = os.path.join(local_model_path, "model.tar.gz")
    s3.download_file(s3_bucket, model_s3_key, model_tar_path)
    print(f"Model downloaded from S3: {model_s3_key}")

    # Extract the model.tar.gz
    with tarfile.open(model_tar_path, "r:gz") as tar:
        tar.extractall(path=local_model_path)

    # Download the min_max_values.json from S3
    min_max_path = os.path.join(local_model_path, "min_max_values.json")
    s3.download_file(s3_bucket, min_max_s3_key, min_max_path)

    # Load the min_max_values.json
    with open(min_max_path, 'r') as f:
        global_vars['min_max_values'] = json.load(f)

    # Load the TensorFlow model
    print(f"Loading model from {local_model_path}")
    model = tf.saved_model.load(local_model_path)
    global_vars['predict'] = model.signatures['serving_default']

    print(f"Model and min-max values loaded from version {highest_version}")

 # Get the highest version folder
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
