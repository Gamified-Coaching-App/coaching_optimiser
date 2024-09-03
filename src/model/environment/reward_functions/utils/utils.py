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

"""
Prepares the data for the subjective parameter forecast model.
    
Steps: 
    Step 1: Extend action variables to 10 days by appending zeros for strength_training and hours_alternative and 
    nr. session = 1 if km_total > 0, else nr. session = 0
    Step 2: Conatenate states and actions along the days dimension: 63 days
    Step 3: Convert to absolute values
    Step 4: Z-score normalise
    Step 5: Slice the last 14 days
    Step 6: Min-max normalise
    Step 7: Add zero padding for subjective parameters for the last 7 days
"""
def prepare_data_subjective_parameter_forecaster(states, actions, min_max_values, standardised_min_max_values):

    original_actions = tf.identity(actions)

    actions = add_missing_variables(actions)

    if actions.shape[1:] != (7, 10):
        raise ValueError(f"Expected actions to have shape [batch, 7, 10], but got {actions.shape}")
    if not tf.reduce_all(tf.equal(actions[:, :, 1:5], original_actions)):
        raise ValueError(f"Variable addition has errors, expected")
    if not tf.reduce_all(tf.equal(actions[:, :, 6:9], tf.zeros_like(actions[:, :, 6:9]))):
        raise ValueError(f"Variables at indices 6-9 are not all zeros")
    
    original_states = tf.identity(states)
    original_actions = tf.identity(actions)
    
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

    actions = z_score_normalize(actions)

    if actions.shape[1:] != (63, 10):
        raise ValueError(f"After z-score normalisation: Expected actions to have shape [batch, 63, 10], but got {actions.shape}")

    actions = actions[:, -14:, :]

    if actions.shape[1:] != (14, 10):
        raise ValueError(f"After slicing 14 days: Expected actions to have shape [batch, 14, 10], but got {actions.shape}")

    actions = min_max_normalize(actions, standardised_min_max_values)
    if actions.shape[1:] != (14, 10):
        raise ValueError(f"After min-max normalise: Expected actions to have shape [batch, 14, 10], but got {actions.shape}")

    original_actions = tf.identity(actions)
    first_7_days_last_3 = actions[:, :7, -3:]
    padding_zeros = tf.zeros_like(first_7_days_last_3)
    padding = tf.concat([first_7_days_last_3, padding_zeros], axis=1)
    first_7_vars_all_days = actions[:, :, :-3]
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

"""
helper function for Step 1
"""
def add_missing_variables(actions):
    nr_sessions = tf.tanh(actions[:, :, 0] * 1000)
    nr_sessions = tf.expand_dims(nr_sessions , axis=2) 
    
    actions = tf.concat([nr_sessions , actions], axis=2)

    if actions.shape[1:] != (7,5):
        raise ValueError(f"Expected actions to have shape [batch, 7, 5], but got {actions.shape}")
    
    padding = tf.zeros([tf.shape(actions)[0], tf.shape(actions)[1], 5], dtype=actions.dtype)
    actions = tf.concat([actions, padding], axis=2)

    if actions.shape[1:] != (7,10):
        raise ValueError(f"Expected actions to have shape [batch, 7, 10], but got {actions.shape}")
    
    return actions

"""
helper function for Step 4
"""
def z_score_normalize(full_sequence):
    mean = tf.reduce_mean(full_sequence, axis=[0, 1], keepdims=True)
    stddev = tf.math.reduce_std(full_sequence, axis=[0, 1], keepdims=True)
    
    normalized_sequence = (full_sequence - mean) / stddev
    
    return normalized_sequence

"""
helper function for Step 3
"""
def convert_to_absolute_values(actions, min_max_values):
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

    absolute_values_tensor = tf.concat([
        nr_sessions, total_km, km_z3_4, km_z5_t1_t2, km_sprinting,
        strength_training, hours_alternative, perceived_exertion,
        perceived_training_success, perceived_recovery
    ], axis=-1)
    
    return absolute_values_tensor

"""
helper function for Step 6
"""
def min_max_normalize(actions, standardised_min_max_values):
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
    
    normalized_values_tensor = tf.stack([
        nr_sessions, total_km, km_z3_4, km_z5_t1_t2, km_sprinting,
        strength_training, hours_alternative, perceived_exertion,
        perceived_training_success, perceived_recovery
    ], axis=-1)
    
    return normalized_values_tensor

"""
function to prepare data for injury model
Steps:
    Step 1: Slice out the last 7 days from the input tensor
    Step 2: Concatenate objective parameters for last 7 days with subjective parameters from inference to subjective parameter forecast model
"""
def prepare_data_injury_model(last_14_days, subjective_params):
    if last_14_days.shape[1:] != (14, 10):
        raise ValueError(f"Expected data to have shape [batch, 14, 10], but got {last_14_days.shape}")
    if subjective_params.shape[1:] != (7, 3):
        raise ValueError(f"Expected subjective_params to have shape [batch, 14, 3], but got {subjective_params.shape}")
    
    last_7_days = last_14_days[:, -7:, :]
    
    last_7_days = tf.concat([
        last_7_days[:, :, :7],   
        subjective_params        
    ], axis=2)
    
    return last_7_days

"""
helper function to convert normalised values to absolute values
"""
def get_absolute_values(data, min_max_values, variable='total km'):
    min = min_max_values[variable]['min']
    max = min_max_values[variable]['max']
    absolute = data * (max - min) + min
    return absolute

"""
global contant parameters for penalty term computation
"""
DAYS_FOR_OVERALL_WORKLOAD_COMPARISON = 28
STEEPNESS_GREATER = 1.0
EPSILON = 0.01
STEEPNESS_COUNT = 50.0

"""
function to have a smooth approximation of GREATER THAN using ReLU
"""
def smooth_greater(smaller, greater):
    return STEEPNESS_GREATER * (tf.nn.relu(greater - smaller))

"""
function to have a smooth approximation of AND using minimum
"""
def smooth_and(condition1, condition2):
    return tf.minimum(condition1, condition2)

""" 
function to enforce hard contraint of ACWR < 1.5
"""
def test_overall_load(states_total_km, actions_total_km):

    avg_states_km_total = tf.reduce_mean(states_total_km[:, -DAYS_FOR_OVERALL_WORKLOAD_COMPARISON:], axis=1)
    avg_actions_km_total = tf.reduce_mean(actions_total_km, axis=1)
    running_progression = avg_actions_km_total / (avg_states_km_total + EPSILON)

    condition_upper_treshold1 = smooth_greater(greater=running_progression, smaller=1.5)
    condition_upper_treshold2 = smooth_greater(greater=avg_states_km_total, smaller=10.0/DAYS_FOR_OVERALL_WORKLOAD_COMPARISON)
    condition_upper_treshold = smooth_and(condition_upper_treshold1,  condition_upper_treshold2)
    condition_running = condition_upper_treshold

    return condition_running