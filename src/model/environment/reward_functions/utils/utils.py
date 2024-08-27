import tensorflow as tf
import pandas as pd
import numpy as np
import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
src_path = os.path.abspath(os.path.join(current_dir, '../../'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

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

def prepare_data_subjective_parameter_forecaster(states, actions, min_max_values, standardised_min_max_values):
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

    original_actions = tf.identity(actions)

    # Add Nr. Sessions, Strength Sessions and Hours Alternative + Subj to actions: result is batch, 7, 10
    actions = add_missing_variables(actions)

    if actions.shape[1:] != (7, 10):
        raise ValueError(f"Expected actions to have shape [batch, 7, 10], but got {actions.shape}")
    if not tf.reduce_all(tf.equal(actions[:, :, 1:5], original_actions)):
        raise ValueError(f"Variable addition has errors, expected")
    if not tf.reduce_all(tf.equal(actions[:, :, 6:9], tf.zeros_like(actions[:, :, 6:9]))):
        raise ValueError(f"Variables at indices 6-9 are not all zeros")
    
    # Save the original states and actions
    original_states = tf.identity(states)
    original_actions = tf.identity(actions)
    
    # Concatenate actions and states along the days dimension
    actions = tf.concat([states, actions], axis=1)

    if actions.shape[1:] != (63, 10):
        raise ValueError(f"Expected actions to have shape [batch, 63, 10], but got {actions.shape}")
    if not tf.reduce_all(tf.equal(actions[:, -7:, :], original_actions)):
        raise ValueError(f"Actions: Tensors wrongly concatenated across days dimension")
    if not tf.reduce_all(tf.equal(actions[:, :-7, :], original_states)):
        raise ValueError(f"States: Tensors wrongly concatenated across days dimension")
    
    actions = convert_to_absolute_values(actions, min_max_values)
    if actions.shape[1:] != (63, 10):
        raise ValueError(f"After conversion to absolute: Expected actions to have shape [batch, 63, 10], but got {actions.shape}")

    # standardise the full sequence
    actions = z_score_normalize(actions)

    if actions.shape[1:] != (63, 10):
        raise ValueError(f"After z-score normalisation: Expected actions to have shape [batch, 63, 10], but got {actions.shape}")

    # Slice the last 14 days from full sequence 
    actions = actions[:, -14:, :]

    if actions.shape[1:] != (14, 10):
        raise ValueError(f"After slicing 14 days: Expected actions to have shape [batch, 14, 10], but got {actions.shape}")

    actions = min_max_normalize(actions, standardised_min_max_values)
    if actions.shape[1:] != (14, 10):
        raise ValueError(f"After min-max normalise: Expected actions to have shape [batch, 14, 10], but got {actions.shape}")

    # # pad actions with zeros for subjective parameters
    # padding = tf.zeros([tf.shape(actions)[0], tf.shape(actions)[1], 3], dtype=actions.dtype)
    # actions = tf.concat([actions, padding], axis=2)

    original_actions = tf.identity(actions)
    # Extract the first 7 days for the last 3 variables
    first_7_days_last_3 = actions[:, :7, -3:]
    # Create padding of zeros with shape (batch_size, 7 days, 3 variables)
    padding_zeros = tf.zeros_like(first_7_days_last_3)
    # Concatenate the first 7 days last 3 variables with the zero padding to create the 14x3 padding
    padding = tf.concat([first_7_days_last_3, padding_zeros], axis=1)
    # Extract the first 7 variables across all days
    first_7_vars_all_days = actions[:, :, :-3]
    # Concatenate the first 7 variables (across all days) with the 14x3 padding
    actions = tf.concat([first_7_vars_all_days, padding], axis=-1)

    if actions.shape[1:] != (14, 10):
        raise ValueError(f"After min-max normalise: Expected actions to have shape [batch, 14, 10], but got {actions.shape}")
    if not tf.reduce_all(tf.equal(actions[:, :7, :], original_actions[:, :7, :])):
        raise ValueError(f"Final step: first 7 days all vars are not the same")
    if not tf.reduce_all(tf.equal(actions[:, -7:, :-3], original_actions[:, -7:, :-3])):
        raise ValueError(f"Final step: last 7 days first 7 vars are not the same")
    if not tf.reduce_all(tf.equal(actions[:, -7:, -3:], tf.zeros_like(actions[:, -7:, -3:]))):
        raise ValueError(f"Final step: last 7 days last 3 vars are not zeros")

    return actions

def add_missing_variables(actions):
    # Step 1: Compute the tanh of the first action column multiplied by 1000
    nr_sessions = tf.tanh(actions[:, :, 0] * 1000)
    nr_sessions = tf.expand_dims(nr_sessions , axis=2)  # Expand dimensions to match the shape for concatenation
    
    # Step 2: Add the tanh_row as the first column
    actions = tf.concat([nr_sessions , actions], axis=2)

    if actions.shape[1:] != (7,5):
        raise ValueError(f"Expected actions to have shape [batch, 7, 5], but got {actions.shape}")
    
    # Step 3: Add zeros padding as the new columns (6,7,8,9) to make data batch, 7, 10
    padding = tf.zeros([tf.shape(actions)[0], tf.shape(actions)[1], 5], dtype=actions.dtype)
    actions = tf.concat([actions, padding], axis=2)

    if actions.shape[1:] != (7,10):
        raise ValueError(f"Expected actions to have shape [batch, 7, 10], but got {actions.shape}")
    
    return actions

def z_score_normalize(full_sequence):
    # Compute the mean and standard deviation for each variable (across the sequence)
    mean = tf.reduce_mean(full_sequence, axis=[0, 1], keepdims=True)
    stddev = tf.math.reduce_std(full_sequence, axis=[0, 1], keepdims=True)
    
    # Z-score normalization
    normalized_sequence = (full_sequence - mean) / stddev
    
    return normalized_sequence

def convert_to_absolute_values(actions, min_max_values):
    # Convert each variable separately using get_absolute_values and expand dims
    nr_sessions = tf.expand_dims(get_absolute_values(actions[:, :, 0], min_max_values, 'nr. sessions'), axis=-1)
    total_km = tf.expand_dims(get_absolute_values(actions[:, :, 1], min_max_values, 'total km'), axis=-1)
    km_z3_4 = tf.expand_dims(get_absolute_values(actions[:, :, 2], min_max_values, 'km Z3-4'), axis=-1)
    km_z5_t1_t2 = tf.expand_dims(get_absolute_values(actions[:, :, 3], min_max_values, 'km Z5-T1-T2'), axis=-1)
    km_sprinting = tf.expand_dims(get_absolute_values(actions[:, :, 4], min_max_values, 'km sprinting'), axis=-1)
    strength_training = tf.expand_dims(get_absolute_values(actions[:, :, 5], min_max_values, 'strength training'), axis=-1)
    hours_alternative = tf.expand_dims(get_absolute_values(actions[:, :, 6], min_max_values, 'hours alternative'), axis=-1)
    perceived_exertion = tf.expand_dims(get_absolute_values(actions[:, :, 7], min_max_values, 'perceived exertion'), axis=-1)
    perceived_training_success = tf.expand_dims(get_absolute_values(actions[:, :, 8], min_max_values, 'perceived trainingSuccess'), axis=-1)
    perceived_recovery = tf.expand_dims(get_absolute_values(actions[:, :, 9], min_max_values, 'perceived recovery'), axis=-1)

    # Concatenate all the absolute values along the last dimension
    absolute_values_tensor = tf.concat([
        nr_sessions, total_km, km_z3_4, km_z5_t1_t2, km_sprinting,
        strength_training, hours_alternative, perceived_exertion,
        perceived_training_success, perceived_recovery
    ], axis=-1)
    
    return absolute_values_tensor

def min_max_normalize(actions, standardised_min_max_values):
    # Normalize each variable individually
    nr_sessions = (actions[:, :, 0] - standardised_min_max_values['nr. sessions']['min']) / (standardised_min_max_values['nr. sessions']['max'] - standardised_min_max_values['nr. sessions']['min'])
    total_km = (actions[:, :, 1] - standardised_min_max_values['total km']['min']) / (standardised_min_max_values['total km']['max'] - standardised_min_max_values['total km']['min'])
    km_z3_4 = (actions[:, :, 2] - standardised_min_max_values['km Z3-4']['min']) / (standardised_min_max_values['km Z3-4']['max'] - standardised_min_max_values['km Z3-4']['min'])
    km_z5_t1_t2 = (actions[:, :, 3] - standardised_min_max_values['km Z5-T1-T2']['min']) / (standardised_min_max_values['km Z5-T1-T2']['max'] - standardised_min_max_values['km Z5-T1-T2']['min'])
    km_sprinting = (actions[:, :, 4] - standardised_min_max_values['km sprinting']['min']) / (standardised_min_max_values['km sprinting']['max'] - standardised_min_max_values['km sprinting']['min'])
    strength_training = (actions[:, :, 5] - standardised_min_max_values['strength training']['min']) / (standardised_min_max_values['strength training']['max'] - standardised_min_max_values['strength training']['min'])
    hours_alternative = (actions[:, :, 6] - standardised_min_max_values['hours alternative']['min']) / (standardised_min_max_values['hours alternative']['max'] - standardised_min_max_values['hours alternative']['min'])
    perceived_exertion = (actions[:, :, 7] - standardised_min_max_values['perceived exertion']['min']) / (standardised_min_max_values['perceived exertion']['max'] - standardised_min_max_values['perceived exertion']['min'])
    perceived_training_success = (actions[:, :, 8] - standardised_min_max_values['perceived trainingSuccess']['min']) / (standardised_min_max_values['perceived trainingSuccess']['max'] - standardised_min_max_values['perceived trainingSuccess']['min'])
    perceived_recovery = (actions[:, :, 9] - standardised_min_max_values['perceived recovery']['min']) / (standardised_min_max_values['perceived recovery']['max'] - standardised_min_max_values['perceived recovery']['min'])
    
    # Concatenate all the normalized variables along the last dimension
    normalized_values_tensor = tf.stack([
        nr_sessions, total_km, km_z3_4, km_z5_t1_t2, km_sprinting,
        strength_training, hours_alternative, perceived_exertion,
        perceived_training_success, perceived_recovery
    ], axis=-1)
    
    return normalized_values_tensor


def prepare_data_injury_model(last_14_days, subjective_params):
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
    if last_14_days.shape[1:] != (14, 10):
        raise ValueError(f"Expected data to have shape [batch, 14, 10], but got {last_14_days.shape}")
    if subjective_params.shape[1:] != (7, 3):
        raise ValueError(f"Expected subjective_params to have shape [batch, 14, 3], but got {subjective_params.shape}")
    
    # Extract last 7 days from the last 14 days
    last_7_days = last_14_days[:, -7:, :]
    
    # Replace the 7th, 8th, and 9th variables with subjective_params
    # subjective_params is assumed to be of shape (batch, 7, 3)
    last_7_days = tf.concat([
        last_7_days[:, :, :7],   
        subjective_params        
    ], axis=2)
    
    return last_7_days

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

# Hard contraints - 100000.0:
    # 1) Regulate overall workload: Comparison actions vs. state 28 days 
    # - average km value (total, Z34, Z5-T1-T2, sprinting) not more than 5 times higher actions vs. state
    # - number of zero km days between 0.25 and 3 times higher in actions vs. state
    # - number of strength training days between 0.2 and 2 times higher in actions vs. state
    # - number of alternative sessions between 0.2 and 2 times higher in actions vs. state
    # 2) Determine workload distribution: Action looking at each non-zero running day in isolation
    # - Z3-4 km between 10-40% of total km
    # - Z5-T1-T2 km between 0-20% of total km
    # - sprinting km between 0-10% of total km
    # Ultra hard contraints - 100000000.0
    # 1) Decline full zero weeks: 10 times higher penalty for full zero weeks
    # - number of zero running days >= 7
    # - number of zero strength training days >= 7
    # - number of zero alternative sessions >= 7
    # 2) Logical errors
    # - number running sessions (total km > 0) + strength training + alternative sessions = nr. sessions
    # - total km = 0 but Z3-4, Z5-T1-T2 or sprinting > 0
    # 3) Avoid very small training days: 
    # - total km < 2km
    # - alternative: hours < 0.5

DAYS_FOR_OVERALL_WORKLOAD_COMPARISON = 28
STEEPNESS_GREATER = 1.0
EPSILON = 0.01
STEEPNESS_COUNT = 50.0

""" Sets values to 1 if second value is GREATER than first value """
def smooth_greater(smaller, greater):
    return STEEPNESS_GREATER * (tf.nn.relu(greater - smaller))

def smooth_equal(x, y):
    condition1 = smooth_greater(smaller=x, greater=y + EPSILON)
    condition2 = smooth_greater(greater=x, smaller=y - EPSILON)
    return tf.minimum(condition1, condition2)

def smooth_not_equal(x, y):
    condition1 = smooth_greater(smaller=x, greater=y + EPSILON)
    condition2 = smooth_greater(greater=x, smaller=y - EPSILON)
    return tf.maximum(condition1, condition2)

def smooth_and(condition1, condition2):
    return tf.minimum(condition1, condition2)

def smooth_count(tensor):
    return STEEPNESS_COUNT * tf.nn.tanh(tensor)

""" Average km per day shall be no more than 2x compared to the last 28 days"""
def test_overall_load(states_total_km, actions_total_km):

    avg_states_km_total = tf.reduce_mean(states_total_km[:, -DAYS_FOR_OVERALL_WORKLOAD_COMPARISON:], axis=1)
    avg_actions_km_total = tf.reduce_mean(actions_total_km, axis=1)
    running_progression = avg_actions_km_total / (avg_states_km_total + EPSILON)

    condition_upper_treshold1 = smooth_greater(greater=running_progression, smaller=1.5)
    condition_upper_treshold2 = smooth_greater(greater=avg_states_km_total, smaller=10.0/DAYS_FOR_OVERALL_WORKLOAD_COMPARISON)
    condition_upper_treshold = smooth_and(condition_upper_treshold1,  condition_upper_treshold2)
    condition_running = condition_upper_treshold

    return condition_running