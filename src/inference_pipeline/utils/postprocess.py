from training_data.training_data import InputData, OutputData
import numpy as np

"""
function to postprocess load targets after inference
"""
def postprocess(output, min_max_values):
    output_object = OutputData(output)   
     
    return convert_to_absolute_values(output_object, min_max_values)

"""
helper function to convert the normalised output to absolute values
"""
def convert_to_absolute_values(output_object, min_max_values):
    print("converting to absolute values...")

    result = []
    variables = [
        'total km',
        'km Z3-4',
        'km Z5-T1-T2',
        'km sprinting'
    ]
    
    for var in variables:
        abs_values = get_absolute_values(output_object[var], min_max_values, var)
        result.append(abs_values)
    
    stacked_result = np.stack(result, axis=-1)
    
    return stacked_result

"""
helper function to perform conversion to absolute values for one variable
"""
def get_absolute_values(data, min_max_values, variable='total km'):
    min = min_max_values[variable]['min']
    max = min_max_values[variable]['max']
    absolute = data * (max - min) + min
    return absolute