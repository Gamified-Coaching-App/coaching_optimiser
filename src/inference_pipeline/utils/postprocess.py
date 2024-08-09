from training_data.training_data import InputData, OutputData
import numpy as np

 # TO-DO: in Postprocessing include conversion from standardised to absolute

def postprocess(input, output):
    input_object = InputData(input)
    result = []
    if(True):
        result = take_rule_based_approach(input_object)
    
    return result

def take_rule_based_approach(input_data_object):
    variables = [
        'numberSessions',
        'kmTotal',
        'kmZ3Z4',
        'kmZ5',
        'kmSprint',
        'numberStrengthSessions',
        'hoursAlternative'
    ]
    
    result = []
    
    # Iterate over batches (users)
    for user_index in range(input_data_object[variables[0]].shape[0]):
        user_result = []
        # Iterate over variables
        for var in variables:
            data_array = input_data_object[var]
            last_7_days = data_array[user_index, -7:]  # Extract last 7 days for the current user and variable
            user_result.append(last_7_days)
        result.append(np.stack(user_result, axis=-1))  # Stack results to shape (days, variables)
    
    return np.stack(result, axis=0)