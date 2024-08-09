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
STEEPNESS_GREATER = 100.0
EPSILON = 1e-6
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
def test_overall_load(states_total_km, actions_total_km, actions_km_Z3_4, states_km_Z3_4, actions_km_Z5_T1_T2, states_km_Z5_T1_T2, actions_km_sprinting, states_km_sprinting, states_hours_alternative, actions_hours_alternative, states_strength_training, actions_strength_training):
    # RUNNING LOAD
    avg_states_km_total = tf.reduce_mean(states_total_km[:, -DAYS_FOR_OVERALL_WORKLOAD_COMPARISON:], axis=1)
    avg_actions_km_total = tf.reduce_mean(actions_total_km, axis=1)
    total_actions_km_total = tf.reduce_sum(actions_total_km, axis=1)
    # avg_states_km_Z3_4 = tf.reduce_mean(states_km_Z3_4[:, -DAYS_FOR_OVERALL_WORKLOAD_COMPARISON:], axis=1)
    # avg_actions_km_Z3_4 = tf.reduce_mean(actions_km_Z3_4, axis=1)
    # avg_states_km_Z5_T1_T2 = tf.reduce_mean(states_km_Z5_T1_T2[:, -DAYS_FOR_OVERALL_WORKLOAD_COMPARISON:], axis=1)
    # avg_actions_km_Z5_T1_T2 = tf.reduce_mean(actions_km_Z5_T1_T2, axis=1)
    # avg_states_km_sprinting = tf.reduce_mean(states_km_sprinting[:, -DAYS_FOR_OVERALL_WORKLOAD_COMPARISON:], axis=1)
    # avg_actions_km_sprinting = tf.reduce_mean(actions_km_sprinting, axis=1)

    condition_upper_treshold1 = smooth_greater(greater=avg_actions_km_total, smaller=avg_states_km_total * 1.5)
    condition_upper_treshold2 = smooth_greater(greater=total_actions_km_total, smaller=3.0)
    condition_upper_treshold = smooth_and(condition_upper_treshold1,  condition_upper_treshold2)
    condition_running = condition_upper_treshold
    # condition_Z3_4 = smooth_greater(greater=avg_actions_km_Z3_4, smaller=avg_states_km_Z3_4 * 1.5)
    # condition_Z5_T1_T2 = smooth_greater(greater=avg_actions_km_Z5_T1_T2, smaller=avg_states_km_Z5_T1_T2 * 1.5)
    # condition_sprinting = smooth_greater(greater=avg_actions_km_sprinting, smaller=avg_states_km_sprinting * 1.5)
    #condition_running = condition_total #+ condition_Z3_4 + condition_Z5_T1_T2 + condition_sprinting

    # ALTERNATIVE LOAD
    # avg_states_alternative = tf.reduce_mean(states_hours_alternative[:, -DAYS_FOR_OVERALL_WORKLOAD_COMPARISON:], axis=1)
    # avg_actions_alternative = tf.reduce_mean(actions_hours_alternative, axis=1) 
    # condition_alt = smooth_greater(greater=avg_actions_alternative, smaller=1.2 * avg_states_alternative)

    # STRENGTH TRAINING LOAD
    # strength_total = tf.reduce_sum(actions_strength_training, axis=1)
    # condition_strength = smooth_greater(smaller=4.0, greater=strength_total)

    return condition_running #+ condition_alt + condition_strength

""" DONE: On running days, Z3-4 km shall be no more than 40%, Z5 no more than 30% and sprint km no more than 10% of total km"""
def test_running_zone_distribution(actions_km_Z3_4, actions_km_Z5_T1_T2, actions_km_sprinting, actions_total_km):
    Z34_upper_threshold_condition = smooth_greater(smaller=0.4 * actions_total_km, greater=actions_km_Z3_4)
    Z5_upper_threshold_condition = smooth_greater(greater=actions_km_Z5_T1_T2, smaller=0.3 * actions_total_km)
    sprint_upper_threshold_condition = smooth_greater(greater=actions_km_sprinting, smaller=0.1 * actions_total_km)
    return tf.reduce_sum(Z34_upper_threshold_condition + Z5_upper_threshold_condition + sprint_upper_threshold_condition, axis=1)

""" DONE: Weeks with no runs, no strength or no alternative sessions shall be penalized """
def test_emtpy_weeks(actions_total_km, actions_hours_alternative, actions_strength_training):
    total_running_km = tf.reduce_sum(actions_total_km, axis=1)
    # zero_strength_days = tf.reduce_sum(actions_strength_training, axis=1)
    # zero_alternative_days = tf.reduce_sum(actions_hours_alternative, axis=1)
    condition1 = smooth_greater(greater=2.0, smaller=total_running_km)
    # condition2 = smooth_greater(greater=EPSILON, smaller=zero_strength_days)
    # condition3 = smooth_greater(greater=EPSILON, smaller=zero_alternative_days)
    return (condition1)#+ condition2 + condition3) * 1000

""" DONE: Logical errors in the suggestions shall be penalized:
    - total number of sessions does not equal running sessions (total km > 0) + strength training + alternative sessions
    - total km = 0 but Z3-4, Z5-T1-T2 or sprinting > 0
 """
def test_logical_errors(actions_nr_sessions, actions_total_km, actions_km_Z3_4, actions_km_Z5_T1_T2, actions_km_sprinting, actions_strength_training, actions_hours_alternative):
    running_sessions = smooth_count(actions_total_km)
    strength_sessions = smooth_count(actions_strength_training)
    alternative_sessions = smooth_count(actions_hours_alternative)
    condition1 = smooth_not_equal(running_sessions + strength_sessions + alternative_sessions, actions_nr_sessions)
    condition2 = smooth_and(smooth_equal(actions_total_km, 0.0), smooth_greater(greater=actions_km_Z3_4, smaller=0.0) + smooth_greater(greater=actions_km_Z5_T1_T2, smaller=0.0) + smooth_greater(greater=actions_km_sprinting, smaller=0.0))
    result = tf.reduce_sum(condition1 + condition2, axis=1)
    return result

""" DONE: Small training days shall be penalized, as no workouts can be constructed out of them """
def test_absolute_bounds(actions_total_km, actions_km_Z3_4, actions_km_Z5_T1_T2, actions_km_sprinting, actions_hours_alternative, actions_strength_training):
    # Total km either 0 or > 2 km
    condition1_total_km = smooth_greater(smaller=actions_total_km, greater=2.0)
    condition2_total_km = smooth_greater(smaller=0.01, greater=actions_total_km)
    condition_total_km = smooth_and(condition1_total_km, condition2_total_km)
    
    # Z3-4 km either 0 or > 0.5 km
    condition1_km_Z34 = smooth_greater(smaller=actions_km_Z3_4, greater=0.5)
    condition2_km_Z34 = smooth_greater(smaller=0.0, greater=actions_km_Z3_4)
    condition_Z34 = smooth_and(condition1_km_Z34, condition2_km_Z34)
    
    # Z5-T1-T2 km either 0 or > 0.5 km
    condition1_km_Z5T1T2 = smooth_greater(smaller=actions_km_Z5_T1_T2, greater=0.5)
    condition2_km_Z5T1T2 = smooth_greater(smaller=0.0, greater=actions_km_Z5_T1_T2)
    condition_Z5T1T2 = smooth_and(condition1_km_Z5T1T2, condition2_km_Z5T1T2)
    
    # Sprinting km either 0 or > 0.25 km
    condition1_km_sprinting = smooth_greater(smaller=actions_km_sprinting, greater=0.25)
    condition2_km_sprinting = smooth_greater(smaller=0.0, greater=actions_km_sprinting)
    condition_sprinting = smooth_and(condition1_km_sprinting, condition2_km_sprinting)
    
    # Alternative hours either 0 or > 0.5 hours
    condition1_alternative = smooth_greater(smaller=actions_hours_alternative, greater=0.5)
    condition2_alternative = smooth_greater(smaller=0.0, greater=actions_hours_alternative)
    condition_alternative = smooth_and(condition1_alternative, condition2_alternative)
    
    # CHECK THIS CONDITION AGAIN Strength training 0-1
    condition1_strength_training = smooth_greater(greater=actions_strength_training, smaller=1.0)
    condition2_strength_training = smooth_greater(greater=0.0, smaller=actions_strength_training)
    condition_strength_training = condition1_strength_training + condition2_strength_training

    # On running days Z1-2 km shall be at least 1.5 km to enable warm up
    condition_Z2 = smooth_and(smooth_greater(greater=actions_total_km, smaller=0.0), smooth_greater(smaller=actions_total_km - actions_km_sprinting - actions_km_Z3_4 - actions_km_Z5_T1_T2, greater=1.5))
    
    return tf.reduce_sum(condition_total_km, axis=1)#tf.reduce_sum(condition_total_km + condition_Z34 + condition_Z5T1T2 + condition_sprinting + condition_strength_training + condition_alternative + condition_Z2, axis=1)
