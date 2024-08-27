import boto3
import requests
import json
import numpy as np
import tensorflow as tf
import sys
import traceback

KEY_MAPPING = {
    'numberSessions': 'nr. sessions',
    'kmTotal': 'total km',
    'kmZ3Z4': 'km Z3-4',
    'kmZ5': 'km Z5-T1-T2',
    'kmSprint': 'km sprinting',
    'numberStrengthSessions': 'strength training',
    'hoursAlternative': 'hours alternative',
    'perceivedExertion': 'perceived exertion',
    'perceivedRecovery': 'perceived recovery',
    'perceivedTrainingSuccess': 'perceived trainingSuccess'
}

def normalise(standardised_data, min_max_values):
    for user_data in standardised_data:
        user_id = user_data['userId']
        for day, day_data in user_data['data'].items():
            for key, value in day_data.items():
                if key in KEY_MAPPING:
                    mapped_key = KEY_MAPPING[key]
                    if mapped_key in min_max_values:
                        min_val = min_max_values[mapped_key]['min']
                        max_val = min_max_values[mapped_key]['max']
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

        normalised_data = normalise(input_data, min_max_values)
        
        reshaped_normalised_data = reshape(normalised_data)

        return tf.convert_to_tensor(reshaped_normalised_data, dtype=tf.float32), user_ids
    except Exception as e:
        exc_type, exc_obj, tb = sys.exc_info()
        lineno = tb.tb_lineno
        filename = tb.tb_frame.f_code.co_filename  # Get the script name

        # Print the error message along with the script name and line number
        print(f"Error in preprocess function!! {str(e)} in {filename} on line {lineno}")