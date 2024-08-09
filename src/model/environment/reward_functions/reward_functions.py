import tensorflow as tf
import pandas as pd
import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
src_path = os.path.abspath(os.path.join(current_dir, '../../../'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from model.environment.reward_functions.utils.utils import *
from training_data.training_data import InputData, OutputData

def get_progress(states, actions, min_max_values):
    states = InputData(states)
    actions = OutputData(actions) 
    
    # Calculate absolute values for each variable
    states_nr_sessions = get_absolute_values(states['nr. sessions'], min_max_values, 'nr. sessions')
    actions_nr_sessions = get_absolute_values(actions['nr. sessions'], min_max_values, 'nr. sessions')

    states_total_km = get_absolute_values(states['total km'], min_max_values,'total km')
    actions_total_km = get_absolute_values(actions['total km'], min_max_values,'total km')

    states_km_Z3_4 = get_absolute_values(states['km Z3-4'], min_max_values, 'km Z3-4')
    actions_km_Z3_4 = get_absolute_values(actions['km Z3-4'], min_max_values, 'km Z3-4')

    states_km_Z5_T1_T2 = get_absolute_values(states['km Z5-T1-T2'], min_max_values, 'km Z5-T1-T2')
    actions_km_Z5_T1_T2 = get_absolute_values(actions['km Z5-T1-T2'], min_max_values, 'km Z5-T1-T2')

    states_km_sprinting = get_absolute_values(states['km sprinting'], min_max_values, 'km sprinting')
    actions_km_sprinting = get_absolute_values(actions['km sprinting'], min_max_values, 'km sprinting')

    states_strength_training = get_absolute_values(states['strength training'], min_max_values, 'strength training')
    actions_strength_training = get_absolute_values(actions['strength training'], min_max_values, 'strength training')

    states_hours_alternative = get_absolute_values(states['hours alternative'], min_max_values, 'hours alternative')
    actions_hours_alternative = get_absolute_values(actions['hours alternative'], min_max_values, 'hours alternative')

    # Calculate means for each variable
    mean_states_nr_sessions = tf.reduce_mean(states_nr_sessions, axis=1)
    mean_actions_nr_sessions = tf.reduce_mean(actions_nr_sessions, axis=1)

    mean_states_total_km = tf.reduce_mean(states_total_km, axis=1)
    mean_actions_total_km = tf.reduce_mean(actions_total_km, axis=1)

    mean_states_km_Z3_4 = tf.reduce_mean(states_km_Z3_4, axis=1)
    mean_actions_km_Z3_4 = tf.reduce_mean(actions_km_Z3_4, axis=1)

    mean_states_km_Z5_T1_T2 = tf.reduce_mean(states_km_Z5_T1_T2, axis=1)
    mean_actions_km_Z5_T1_T2 = tf.reduce_mean(actions_km_Z5_T1_T2, axis=1)

    mean_states_km_sprinting = tf.reduce_mean(states_km_sprinting, axis=1)
    mean_actions_km_sprinting = tf.reduce_mean(actions_km_sprinting, axis=1)

    mean_states_strength_training = tf.reduce_mean(states_strength_training, axis=1)
    mean_actions_strength_training = tf.reduce_mean(actions_strength_training, axis=1)

    mean_states_hours_alternative = tf.reduce_mean(states_hours_alternative, axis=1)
    mean_actions_hours_alternative = tf.reduce_mean(actions_hours_alternative, axis=1)

    epsilon = 0.01
    # Calculate the deltas for each variable
    delta_nr_sessions = mean_actions_nr_sessions - mean_states_nr_sessions
    delta_total_km = mean_actions_total_km / ( mean_states_total_km + epsilon)
    delta_km_Z3_4 = mean_actions_km_Z3_4 - mean_states_km_Z3_4
    delta_km_Z5_T1_T2 = mean_actions_km_Z5_T1_T2 - mean_states_km_Z5_T1_T2
    delta_km_sprinting = mean_actions_km_sprinting - mean_states_km_sprinting
    delta_strength_training = mean_actions_strength_training - mean_states_strength_training
    delta_hours_alternative = mean_actions_hours_alternative - mean_states_hours_alternative

    # Sum the deltas to get the total progress
    # total_progress = (delta_nr_sessions + delta_total_km + delta_km_Z3_4 + 
    #                   delta_km_Z5_T1_T2 + delta_km_sprinting + 
    #                   delta_strength_training + delta_hours_alternative)

    return delta_total_km

def get_injury_score(self, states, actions):
    data = prepare_data_subjective_parameter_forecaster(states, actions)
    full_data = self.predict_subjective_parameters(data)['output_0']
    df = prepare_data_injury_model(full_data)
   
    all_predictions = []
    for model in self.injury_models:
        predictions = model.predict_proba(df)[: ,1].astype('float32')
        all_predictions.append(predictions)

    all_predictions_tensor = tf.stack(all_predictions, axis=0)
    mean_predictions = tf.reduce_mean(all_predictions_tensor, axis=0)
    return mean_predictions

def get_hard_penalty(states, actions, min_max_values):
    HARD_PENALTY = 10000.0
    states = InputData(states)
    actions = OutputData(actions) 

    states_nr_sessions = get_absolute_values(states['nr. sessions'], min_max_values, 'nr. sessions')
    actions_nr_sessions = get_absolute_values(actions['nr. sessions'], min_max_values, 'nr. sessions')

    states_total_km = get_absolute_values(states['total km'], min_max_values,'total km')
    actions_total_km = get_absolute_values(actions['total km'], min_max_values,'total km')

    states_km_Z3_4 = get_absolute_values(states['km Z3-4'], min_max_values, 'km Z3-4')
    actions_km_Z3_4 = get_absolute_values(actions['km Z3-4'], min_max_values, 'km Z3-4')

    states_km_Z5_T1_T2 = get_absolute_values(states['km Z5-T1-T2'], min_max_values, 'km Z5-T1-T2')
    actions_km_Z5_T1_T2 = get_absolute_values(actions['km Z5-T1-T2'], min_max_values, 'km Z5-T1-T2')

    states_km_sprinting = get_absolute_values(states['km sprinting'], min_max_values, 'km sprinting')
    actions_km_sprinting = get_absolute_values(actions['km sprinting'], min_max_values, 'km sprinting')

    states_strength_training = get_absolute_values(states['strength training'], min_max_values, 'strength training')
    actions_strength_training = get_absolute_values(actions['strength training'], min_max_values, 'strength training')

    states_hours_alternative = get_absolute_values(states['hours alternative'], min_max_values, 'hours alternative')
    actions_hours_alternative = get_absolute_values(actions['hours alternative'], min_max_values, 'hours alternative')

    condition_overall_load = test_overall_load(states_total_km, actions_total_km, actions_km_Z3_4, states_km_Z3_4, actions_km_Z5_T1_T2, states_km_Z5_T1_T2, actions_km_sprinting, states_km_sprinting, states_hours_alternative, actions_hours_alternative, states_strength_training, actions_strength_training)
    condition_running_zone_distribution = test_running_zone_distribution(actions_km_Z3_4, actions_km_Z5_T1_T2, actions_km_sprinting, actions_total_km)
    condition_empty_weeks = test_emtpy_weeks(actions_total_km, actions_hours_alternative, actions_strength_training)

    # condition_logical_errors = test_logical_errors(actions_nr_sessions, actions_total_km, actions_km_Z3_4, actions_km_Z5_T1_T2, actions_km_sprinting, actions_strength_training, actions_hours_alternative)
    
    condition_absolute_bounds = test_absolute_bounds(actions_total_km, actions_km_Z3_4, actions_km_Z5_T1_T2, actions_km_sprinting, actions_hours_alternative, actions_strength_training)
    
    HARD_PENALTY = 10000.0

    conditions = condition_absolute_bounds #+ condition_overall_load + condition_empty_weeks  + condition_running_zone_distribution
    
    penalties = conditions * HARD_PENALTY
    return penalties