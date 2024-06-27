import os
import json
import boto3
import numpy as np
import tensorflow as tf

# Define global variables to hold the min-max scaling parameters
min_max = None

def load_min_max_params():
    global min_max
    if min_max is None:
        # Load the min_max.json from S3
        s3 = boto3.client('s3')
        bucket_name = 'blazemodelregistry'
        min_max_key = 'optimizer/latest/min_max_values.json'

        min_max_path = '/tmp/min_max_values.json'
        s3.download_file(bucket_name, min_max_key, min_max_path)
        with open(min_max_path, 'r') as f:
            min_max = json.load(f)

# Preprocess the incoming data
def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def preprocess(input_data):
    # Load min_max_values.json if not already loaded
    load_min_max_params()
    # Dummy preprocessing, return the input data as-is
    return input_data

# Make inference
def predict_fn(input_data):
    # Use a dummy input for inference testing
    keras_data = np.ones(2, 21, 10).astype(np.float32)
    predict = model.signatures['serving_default']
    predictions = predict(tf.constant(keras_data))['output_0'].numpy()
    return predictions

# Postprocess the data
def output_fn(prediction, content_type):
    if content_type == 'application/json':
        return json.dumps(prediction.tolist())
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def handler(event, context):
    global model
    # Load the model from the default SageMaker model directory
    if model is None:
        model_dir = '/opt/ml/model'
        model = tf.saved_model.load(model_dir)
    
    # Get the request data
    request_body = event['body']
    request_content_type = event['headers']['Content-Type']
    
    # Preprocess the data
    input_data = input_fn(request_body, request_content_type)
    print("Input data arrived:", input_data)
    preprocessed_data = preprocess(input_data)
    
    # Make predictions
    prediction = predict_fn(preprocessed_data)
    
    # Postprocess the data
    response_content_type = event['headers']['Accept']
    response_body = output_fn(prediction, response_content_type)
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': response_content_type
        },
        'body': response_body
    }
