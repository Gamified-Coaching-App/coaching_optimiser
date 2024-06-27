import os
import json
import boto3
import numpy as np
import joblib
from sagemaker import get_execution_role
import keras 

# Define global variables to hold the model and min-max scaling parameters
model = None
min_max = None

def load_model_and_params(model_dir):
    global model, min_max
    if model is None or min_max is None:
        # Load the model
        model = joblib.load(os.path.join(model_dir, "model.joblib"))
        
        # Load the min_max.json
        s3 = boto3.client('s3')
        bucket_name = 'blazemodelregistry'
        key = os.path.join(model_dir, "min_max.json")
        s3.download_file(bucket_name, key, '/tmp/min_max_values.json')
        with open('/tmp/min_max_values.json', 'r') as f:
            min_max = json.load(f)

# Preprocess the incoming data
def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def preprocess(input_data):
    global min_max
    # # Apply min-max scaling to the input data
    return input_data

# Make inference
def predict_fn(input_data):
    global model
    predictions = model.predict(input_data)
    return predictions

# Postprocess the data
def output_fn(prediction, content_type):
    if content_type == 'application/json':
        return json.dumps(prediction.tolist())
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def handler(event, context):
    # Load the model and min_max.json if not already loaded
    model_dir = os.environ['MODEL_DIR']
    load_model_and_params(model_dir)
    
    # Get the request data
    request_body = event['body']
    request_content_type = event['headers']['Content-Type']
    
    # Preprocess the data
    input_data = input_fn(request_body, request_content_type)
    print("Input data arrived", input_data)
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
