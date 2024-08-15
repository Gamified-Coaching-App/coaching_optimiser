import boto3
import requests
import json
import numpy as np
import tensorflow as tf
import sys
import traceback


def get_mean_std(dynamo_db_client, user_ids):
    mean_std_dict = {}

    for user_id in user_ids:
        try:
            response = dynamo_db_client.get_item(
                TableName='coaching_mean_stdv',
                Key={'userId': {'S': user_id}}
            )
            if 'Item' in response:
                user_data = response['Item']
                mean_std_dict[user_id] = {key: json.loads(value['S']) for key, value in user_data.items() if key != 'userId'}
            else:
                raise KeyError

        except KeyError:
            requests.post('http://Coachi-Coach-bgtKlzJd2GCw-908383528.eu-west-2.elb.amazonaws.com/updatemeanstdv', json={'userId': user_id})
            response = dynamo_db_client.get_item(
                TableName='coaching_mean_stdv',
                Key={'userId': {'S': user_id}}
            )
            user_data = response['Item']
            mean_std_dict[user_id] = {key: json.loads(value['S']) for key, value in user_data.items() if key != 'userId'}

    return mean_std_dict

def standardise(input_data, min_max_values):
    for user_data in input_data:
        user_id = user_data['userId']
        for day, day_data in user_data['data'].items():
            for key, value in day_data.items():
                if key in min_max_values[user_id]:
                    mean = min_max_values[user_id][key]['mean']
                    stdv = min_max_values[user_id][key]['stdv']
                    if stdv != 0:
                        day_data[key] = (value - mean) / stdv
                    else:
                        day_data[key] = value
    return input_data

def normalise(standardised_data, min_max_values):
    for user_data in standardised_data:
        user_id = user_data['userId']
        for day, day_data in user_data['data'].items():
            for key, value in day_data.items():
                if key in min_max_values:
                    min_val = min_max_values[key]['min']
                    max_val = min_max_values[key]['max']
                    day_data[key] = (value - min_val) / (max_val - min_val)
    return standardised_data

def reshape(normalized_data, days=56):
    feature_order = [
        'numberSessions', 'kmTotal', 'kmZ3Z4', 'kmZ5', 'kmSprint',
        'numberStrengthSessions', 'hoursAlternative', 'perceivedExertion',
        'perceivedRecovery', 'perceivedTrainingSuccess'
    ]
    
    batch_data = []
    for user_data in normalized_data:
        user_batch = []
        for day in range(1, days + 1):
            day_key = f'day{day}'
            day_data = [user_data['data'][day_key][feature] for feature in feature_order]
            user_batch.append(day_data)
        batch_data.append(user_batch)
    
    return np.array(batch_data)

def preprocess(input_data, min_max_values):
    try:
        user_ids = [data['userId'] for data in input_data]

        reshaped_raw_data = reshape(input_data)

        # Normalize data
        normalised_data = normalise(input_data, min_max_values)
        
        # Reshape data
        reshaped_normalised_data = reshape(normalised_data)

        return tf.convert_to_tensor(reshaped_normalised_data, dtype=tf.float32), user_ids
    except Exception as e:
        exc_type, exc_obj, tb = sys.exc_info()
        lineno = tb.tb_lineno
        filename = tb.tb_frame.f_code.co_filename  # Get the script name

        # Print the error message along with the script name and line number
        print(f"Error in preprocess function!! {str(e)} in {filename} on line {lineno}")