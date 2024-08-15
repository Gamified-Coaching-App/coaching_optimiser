from training_data.training_data import InputData, OutputData
import numpy as np

def postprocess(output, min_max_values):
    output_object = OutputData(output)   
     
    return convert_to_absolute_values(output_object, min_max_values)

def convert_to_absolute_values(output_object, min_max_values):
    result = []
    variables = [
       'nr. sessions',
        'total km',
        'km Z3-4',
        'km Z5-T1-T2',
        'km sprinting',
        'strength training',
        'hours alternative'
    ]
    
    for var in variables:
        # Assuming get_absolute_values returns an array of shape (batch, 7)
        abs_values = get_absolute_values(output_object[var], min_max_values, var)
        result.append(abs_values)
    
    # Stack the results along a new last axis to get shape (batch, 7, 7)
    stacked_result = np.stack(result, axis=-1)
    
    return stacked_result

def get_absolute_values(data, min_max_values, variable='total km'):
    """
    Extracts and returns the absolute values of the specified variable from the data for all batches,
    Input has shape [batch_size, time_steps]
    Output has shape [batch_size, time_steps]
    """
    min = min_max_values[variable]['min']
    max = min_max_values[variable]['max']
    absolute = data * (max - min) + min
    return absolute

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