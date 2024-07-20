import tensorflow as tf
import pandas as pd
import numpy as np

def get_variable(data, variable='km total'):
    """
    Extracts and returns the specified variable from the data for all batches,
    Input has shape [batch_size, time_steps, num_features]
    Output has shape [batch_size, time_steps]
    """
    variable_index = get_var_index()[variable]  # Get index for the variable

    return data[:, :, variable_index]

def get_var_index():
    """
    Return a dictionary of variables to their respective indices.
    """
    return {
        'nr sessions': 0,
        'km total': 1,
        'km Z3-Z4': 2,
        'km Z5': 3,
        'km sprint': 4,
        'nr strength': 5,
        'hours alternative': 6,
        'exertion': 7,
        'recovery': 8,
        'training success': 9
    }

def get_day(data, day=0):
    return data[day, :]

def prepare_data_subjective_parameter_forecaster(states, actions):
    """
    Prepares the data for the subjective parameter forecast model.
    
    Parameters:
        states (tf.Tensor): Tensor of shape [samples, 10, 21] representing state variables over 21 days.
        actions (tf.Tensor): Tensor of shape [samples, 7, 7] representing action variables over 7 days.
    
    Steps: 
        Step 1: Extend actions to 10 days by appending 3 rows of zeros
        Step 2: Slice the last 7 days from states
        Step 3: Concatenate states and actions along the days dimension

    Returns:
        tf.Tensor: Tensor of shape [samples, 10, 14] after processing.
    """
    # Extend actions to 10 days by appending 3 columns of zeros
    zero_padding_days = tf.zeros([tf.shape(actions)[0], tf.shape(actions)[1], 3], dtype=actions.dtype)
    actions_padded = tf.concat([actions, zero_padding_days], axis=2)
    # Slice the last 7 days from states
    states_sliced = states[:, -7:, :]  # Take the last 7 rows
    # Concatenate states and actions along the variable dimension (vertical stack)
    combined_data = tf.concat([actions_padded, states_sliced], axis=1)

    return combined_data

def prepare_data_injury_model(data):
    """
    Prepares the data for the injury model by first slicing out the last 7 days from the input tensor,
    then reordering the data to concatenate each variable's data across these days into a single vector per sample,
    and finally converting the result into a pandas DataFrame with detailed column names.

    Parameters:
        data (tf.Tensor): Input tensor with shape [batch, 14, 10], where 'batch' is the batch size,
                          '14' is the number of days, and '10' is the number of variables per day.

    Returns:
        pd.DataFrame: DataFrame with shape [batch, 70], where each column is named according to the
                      variable and day it represents.
    """
    sliced_data = data[:, -7:, :]
    reshaped_data = tf.reshape(sliced_data, [sliced_data.shape[0], -1])

    variable_list = ['nr. sessions', 'total km', 'km Z3-4', 'km Z5-T1-T2', 'km sprinting',
                     'strength training', 'hours alternative', 'perceived exertion',
                     'perceived trainingSuccess', 'perceived recovery']

    column_names = [f"{var}.{i}" for i in range(7) for var in variable_list]

    df = pd.DataFrame(reshaped_data.numpy(), columns=column_names)

    return df
