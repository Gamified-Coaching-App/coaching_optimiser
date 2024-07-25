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

def get_progress(states, actions):
    states_total_km = get_variable(states, 'total km')
    actions_total_km = get_variable(actions, 'total km')
    sum_states_km_total = tf.reduce_sum(states_total_km, axis=1)
    sum_actions_km_total = tf.reduce_sum(actions_total_km, axis=1)
    progress_per_sample = sum_actions_km_total - sum_states_km_total
    return progress_per_sample

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
    # states = InputData(states)
    # actions = OutputData(actions) 

    # states_nr_sessions = get_absolute_values(states['nr. sessions'], min_max_values, 'nr. sessions')
    # actions_nr_sessions = get_absolute_values(actions['nr. sessions'], min_max_values, 'nr. sessions')

    # states_total_km = get_absolute_values(states['total km'], min_max_values,'total km')
    # actions_total_km = get_absolute_values(actions['total km'], min_max_values,'total km')

    # states_km_Z3_4 = get_absolute_values(states['km Z3-4'], min_max_values, 'km Z3-4')
    # actions_km_Z3_4 = get_absolute_values(actions['km Z3-4'], min_max_values, 'km Z3-4')

    # states_km_Z5_T1_T2 = get_absolute_values(states['km Z5-T1-T2'], min_max_values, 'km Z5-T1-T2')
    # actions_km_Z5_T1_T2 = get_absolute_values(actions['km Z5-T1-T2'], min_max_values, 'km Z5-T1-T2')

    # states_km_sprinting = get_absolute_values(states['km sprinting'], min_max_values, 'km sprinting')
    # actions_km_sprinting = get_absolute_values(actions['km sprinting'], min_max_values, 'km sprinting')

    # states_strength_training = get_absolute_values(states['strength training'], min_max_values, 'strength training')
    # actions_strength_training = get_absolute_values(actions['strength training'], min_max_values, 'strength training')

    # states_hours_alternative = get_absolute_values(states['hours alternative'], min_max_values, 'hours alternative')
    # actions_hours_alternative = get_absolute_values(actions['hours alternative'], min_max_values, 'hours alternative')

    #penalties_list = [
        #test_running_load_increase(states_total_km, actions_total_km, actions_km_Z3_4, states_km_Z3_4, actions_km_Z5_T1_T2, states_km_Z5_T1_T2, actions_km_sprinting, states_km_sprinting),
        #test_zero_km_days(states_total_km, actions_total_km),
        # test_number_strength_training_days(states_strength_training, actions_strength_training),
        # test_number_strength_training_sessions(actions_strength_training),
        # test_number_alternative_sessions(states_hours_alternative, actions_hours_alternative),
        # test_total_alternative_session_volume(states_hours_alternative, actions_hours_alternative),
        # test_z3_4_km_distribution(actions_km_Z3_4, actions_total_km),
        # test_z5_t1_t2_km_distribution(actions_km_Z5_T1_T2, actions_total_km),
        # test_sprinting_km_distribution(actions_km_sprinting, actions_total_km),
        # test_full_zero_weeks(actions_total_km, actions_hours_alternative, actions_strength_training),
        # test_logical_errors(actions_nr_sessions, actions_total_km, actions_km_Z3_4, actions_km_Z5_T1_T2, actions_km_sprinting, actions_strength_training, actions_hours_alternative),
        # test_too_small_training_days(actions_total_km, actions_hours_alternative)
    #]

    # for i, penalty in enumerate(penalties_list):
    #     tf.print(f"Shape of penalty {i}: {tf.shape(penalty)}")

    # penalties = test_number_strength_training_sessions(actions_strength_training, states_strength_training) * HARD_PENALTY

    # return penalties
    HARD_PENALTY = 10000.0

    states_total_km = get_variable(states, 'km total')
    actions_total_km = get_variable(actions, 'km total')
    avg_states_km_total = tf.reduce_mean(states_total_km, axis=1)
    avg_actions_km_total = tf.reduce_mean(actions_total_km, axis=1)

    condition = avg_actions_km_total > 2 * avg_states_km_total

    penalties = tf.where(condition,
                         tf.fill(dims=tf.shape(condition), value=float(HARD_PENALTY)),  
                         tf.fill(dims=tf.shape(condition), value=-0.1))
    
    return penalties