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

""" Average km per day shall be no more than 2x compared to the last 28 days"""
def test_running_load_increase(states_total_km, actions_total_km, actions_km_Z3_4, states_km_Z3_4, actions_km_Z5_T1_T2, states_km_Z5_T1_T2, actions_km_sprinting, states_km_sprinting):
    avg_states_km_total = tf.reduce_mean(states_total_km[:, -DAYS_FOR_OVERALL_WORKLOAD_COMPARISON:], axis=1)
    avg_actions_km_total = tf.reduce_mean(actions_total_km, axis=1)

    avg_states_km_Z3_4 = tf.reduce_mean(states_km_Z3_4[:, -DAYS_FOR_OVERALL_WORKLOAD_COMPARISON:], axis=1)
    avg_actions_km_Z3_4 = tf.reduce_mean(actions_km_Z3_4, axis=1)

    avg_states_km_Z5_T1_T2 = tf.reduce_mean(states_km_Z5_T1_T2[:, -DAYS_FOR_OVERALL_WORKLOAD_COMPARISON:], axis=1)
    avg_actions_km_Z5_T1_T2 = tf.reduce_mean(actions_km_Z5_T1_T2, axis=1)

    avg_states_km_sprinting = tf.reduce_mean(states_km_sprinting[:, -DAYS_FOR_OVERALL_WORKLOAD_COMPARISON:], axis=1)
    avg_actions_km_sprinting = tf.reduce_mean(actions_km_sprinting, axis=1)

    condition_total = tf.cast(avg_actions_km_total > 5 * avg_states_km_total, tf.float32)
    condition_Z3_4 = tf.cast(avg_actions_km_Z3_4 > 5 * avg_states_km_Z3_4, tf.float32)
    condition_Z5_T1_T2 = tf.cast(avg_actions_km_Z5_T1_T2 > 5 * avg_states_km_Z5_T1_T2, tf.float32)
    condition_sprinting = tf.cast(avg_actions_km_sprinting > 5 * avg_states_km_sprinting, tf.float32)
    result = condition_total + condition_Z3_4 + condition_Z5_T1_T2 + condition_sprinting
    return result

""" Number of running rest days (zero km) shall be no more than 3x and no less than 0.25x compared to the last 28 days """
def test_zero_km_days(states_total_km, actions_total_km):
    zero_days_states = tf.reduce_mean(tf.cast(states_total_km[ : , -DAYS_FOR_OVERALL_WORKLOAD_COMPARISON : ] == 0, tf.float32), axis=1)
    zero_days_actions = tf.reduce_mean(tf.cast(actions_total_km == 0, tf.float32), axis=1)
    condition = (zero_days_actions > 3 * zero_days_states) | (zero_days_actions < 0.25 * zero_days_states)
    return tf.cast(condition, tf.float32)

""" Number of strength training days shall be no more than 2x and no less then 0.2x compared to the past 28 days """
def test_number_strength_training_days(states_strength_training, actions_strength_training):
    avg_states_strength_training = tf.reduce_mean(tf.cast(states_strength_training[:, -DAYS_FOR_OVERALL_WORKLOAD_COMPARISON:] != 0, tf.float32), axis=1)
    avg_actions_strength_training = tf.reduce_mean(tf.cast(actions_strength_training != 0, tf.float32), axis=1)
    condition = (avg_actions_strength_training > 2 * avg_states_strength_training) | (avg_actions_strength_training < 0.2 * avg_states_strength_training)
    return tf.cast(condition, tf.float32)

""" Number of strength sessions per day shall be at max 1 """
def test_number_strength_training_sessions(actions_strength_training, states_strength_training):
    # condition = actions_strength_training > 1
    # penalties = tf.where(condition, tf.ones_like(actions_strength_training), tf.zeros_like(actions_strength_training))
    # return tf.reduce_sum(penalties, axis=1)
    avg_actions_km_total = tf.reduce_mean(actions_strength_training, axis=1)
    avg_states_km_total = tf.reduce_mean(states_strength_training, axis=1)

    condition = avg_actions_km_total > 2 * avg_states_km_total

    penalties = tf.where(condition,
                         tf.fill(dims=tf.shape(condition), value=float(1)),  
                         tf.fill(dims=tf.shape(condition), value=0.0))
    return penalties

""" Number of alternative sessions (hours alternative > 0) shall be no more than 2x and no less than 0.2x compared to the past 28 days """
def test_number_alternative_sessions(states_hours_alternative, actions_hours_alternative):
    avg_states_alternative = tf.reduce_mean(tf.cast(states_hours_alternative[:, -DAYS_FOR_OVERALL_WORKLOAD_COMPARISON:] != 0, tf.float32), axis=1)
    avg_actions_alternative = tf.reduce_mean(tf.cast(actions_hours_alternative != 0, tf.float32), axis=1) 
    condition = (avg_actions_alternative > 2 *  avg_states_alternative) | (avg_actions_alternative < 0.1 * avg_states_alternative)
    return tf.cast(condition, tf.float32)

""" Total alternative session volumne shall be no more than 2x and no less than 0.2x compared to the past 28 days """
def test_total_alternative_session_volume(states_hours_alternative, actions_hours_alternative):
    avg_states_alternative = tf.reduce_mean(states_hours_alternative[:, -DAYS_FOR_OVERALL_WORKLOAD_COMPARISON:], axis=1)
    avg_actions_alternative = tf.reduce_mean(actions_hours_alternative, axis=1) 
    condition = (avg_actions_alternative > 2 *  avg_states_alternative) | (avg_actions_alternative < 0.2 * avg_states_alternative)
    return tf.cast(condition, tf.float32)

""" On running days, Z3-4 km shall be between 10-40% of total km """
def test_z3_4_km_distribution(actions_km_Z3_4, actions_total_km):
    non_zero_km_condition = actions_total_km != 0
    below_threshold_condition = actions_km_Z3_4 < 0.1 * actions_total_km
    above_threshold_condition = actions_km_Z3_4 > 0.4 * actions_total_km
    condition = non_zero_km_condition & (below_threshold_condition | above_threshold_condition)
    # Sum to get the number of days that do not meet the condition
    return tf.reduce_sum(tf.cast(condition, tf.float32), axis=1)

""" On running days, Z5 km shall no more than 30% of total km """
def test_z5_t1_t2_km_distribution(actions_km_Z5_T1_T2, actions_total_km):
    non_zero_km_condition = actions_total_km != 0
    above_threshold_condition = actions_km_Z5_T1_T2 > 0.3 * actions_total_km
    condition = non_zero_km_condition & above_threshold_condition
    # Sum to get the number of days that do not meet the condition
    return tf.reduce_sum(tf.cast(condition, tf.float32), axis=1)

""" On running days, sprint km shall no more than 10% of total km """
def test_sprinting_km_distribution(actions_km_sprinting, actions_total_km):
    non_zero_km_condition = actions_total_km != 0
    above_threshold_condition = actions_km_sprinting > 0.1 * actions_total_km
    condition = non_zero_km_condition & above_threshold_condition
    # Sum to get the number of days that do not meet the condition
    return tf.reduce_sum(tf.cast(condition, tf.float32), axis=1)

""" Weeks with no runs, no strength or no alternative sessions shall be penalized """
def test_full_zero_weeks(actions_total_km, actions_hours_alternative, actions_strength_training):
    zero_running_days = tf.reduce_sum(tf.cast(actions_total_km == 0, tf.float32), axis=1)
    zero_strength_days = tf.reduce_sum(tf.cast(actions_strength_training == 0, tf.float32), axis=1)
    zero_alternative_days = tf.reduce_sum(tf.cast(actions_hours_alternative == 0, tf.float32), axis=1)
    condition = (zero_running_days >= 7) | (zero_strength_days >= 7) | (zero_alternative_days >= 7)
    return tf.cast(condition, tf.float32) * 100

""" Logical errors in the suggestions shall be penalized:
    - total number of sessions does not equal running sessions (total km > 0) + strength training + alternative sessions
    - total km = 0 but Z3-4, Z5-T1-T2 or sprinting > 0
    - total km > 0 and Z1-2 km < 1.5 km: No warmup possible
 """
def test_logical_errors(actions_nr_sessions, actions_total_km, actions_km_Z3_4, actions_km_Z5_T1_T2, actions_km_sprinting, actions_strength_training, actions_hours_alternative):
    running_sessions = tf.cast(actions_total_km > 0, tf.float32)
    strength_sessions = actions_strength_training
    alternative_sessions = tf.cast(actions_hours_alternative > 0, tf.float32)
    condition1 = tf.cast(running_sessions + strength_sessions + alternative_sessions != actions_nr_sessions, tf.float32)
    condition2 = tf.cast(tf.logical_and(actions_total_km == 0, (actions_km_Z3_4 > 0) | (actions_km_Z5_T1_T2 > 0) | (actions_km_sprinting > 0)), tf.float32)
    condition3 = tf.cast(tf.logical_and(actions_total_km > 0, (actions_total_km - actions_km_sprinting - actions_km_Z3_4 - actions_km_Z5_T1_T2) < 1.5), tf.float32)
    result = tf.reduce_sum(condition1 + condition2 + condition3, axis=1)
    return result

""" Small training days shall be penalized, as no workouts can be constructed out of them """
def test_too_small_training_days(actions_total_km, actions_hours_alternative):
    condition1 = tf.cast(actions_total_km < 2, tf.float32)
    condition2 = tf.cast(actions_hours_alternative < 0.5, tf.float32)
    result = tf.reduce_sum(condition1 + condition2, axis=1)
    return result

