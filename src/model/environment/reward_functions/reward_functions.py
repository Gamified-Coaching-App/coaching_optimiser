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

    mean_states_total_km = tf.reduce_mean(states_total_km[:, -28:], axis=1)
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
    progress = mean_actions_total_km / ( mean_states_total_km + epsilon)
    # delta_km_Z3_4 = mean_actions_km_Z3_4 - mean_states_km_Z3_4
    # delta_km_Z5_T1_T2 = mean_actions_km_Z5_T1_T2 - mean_states_km_Z5_T1_T2
    # delta_km_sprinting = mean_actions_km_sprinting - mean_states_km_sprinting
    # delta_strength_training = mean_actions_strength_training - mean_states_strength_training
    # delta_hours_alternative = mean_actions_hours_alternative - mean_states_hours_alternative

    # Sum the deltas to get the total progress
    # total_progress = (delta_nr_sessions + delta_total_km + delta_km_Z3_4 + 
    #                   delta_km_Z5_T1_T2 + delta_km_sprinting + 
    #                   delta_strength_training + delta_hours_alternative)
        
    print("Median progress:", np.median(progress.numpy()))
    tf.print("Mean progress:", tf.reduce_mean(progress))

    return tf.nn.tanh(progress * 2.0) * 100

def get_injury_score(self, states, actions):
    data = prepare_data_subjective_parameter_forecaster(states, actions)
    data = self.predict_subjective_parameters(data)['output_0']
    data = prepare_data_injury_model(data)
    injury_scores = self.predict_injury_risk(data)['output_0']

    return injury_scores

def get_hard_penalty(states, actions, min_max_values, epoch):
    if epoch < 10:
        HARD_PENALTY = 1.0
    else:
        HARD_PENALTY = epoch

    states = InputData(states)
    actions = OutputData(actions)

    states_total_km = get_absolute_values(states['total km'], min_max_values,'total km')
    actions_total_km = get_absolute_values(actions['total km'], min_max_values,'total km')

    states_strength_training = get_absolute_values(states['strength training'], min_max_values, 'strength training')
    actions_strength_training = get_absolute_values(actions['strength training'], min_max_values, 'strength training')

    states_hours_alternative = get_absolute_values(states['hours alternative'], min_max_values, 'hours alternative')
    actions_hours_alternative = get_absolute_values(actions['hours alternative'], min_max_values, 'hours alternative')

    condition = test_overall_load(states_total_km, actions_total_km, states_strength_training, actions_strength_training, states_hours_alternative, actions_hours_alternative)

    penalties = condition * HARD_PENALTY
    
    tf.print("Percentage of compliance:", tf.round((tf.reduce_sum(tf.cast(tf.equal(condition, 0), tf.float32)) / tf.size(condition, out_type=tf.float32)) * 100 * 100) / 100, "%")

    return penalties