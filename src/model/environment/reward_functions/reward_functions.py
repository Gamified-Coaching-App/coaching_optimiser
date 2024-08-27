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

    # Calculate absolute values for all metrics
    states_total_km = get_absolute_values(states['total km'], min_max_values, 'total km')
    actions_total_km = get_absolute_values(actions['total km'], min_max_values, 'total km')

    states_km_Z34 = get_absolute_values(states['km Z3-4'], min_max_values, 'km Z3-4')
    actions_km_Z34 = get_absolute_values(actions['km Z3-4'], min_max_values, 'km Z3-4')

    states_km_Z5_T1_T2 = get_absolute_values(states['km Z5-T1-T2'], min_max_values, 'km Z5-T1-T2')
    actions_km_Z5_T1_T2 = get_absolute_values(actions['km Z5-T1-T2'], min_max_values, 'km Z5-T1-T2')

    states_km_sprinting = get_absolute_values(states['km sprinting'], min_max_values, 'km sprinting')
    actions_km_sprinting = get_absolute_values(actions['km sprinting'], min_max_values, 'km sprinting')

    # Compute mean values for the last 28 days for states and overall for actions
    mean_states_total_km = tf.reduce_mean(states_total_km[:, -28:], axis=1)
    mean_actions_total_km = tf.reduce_mean(actions_total_km, axis=1)

    mean_states_km_Z34 = tf.reduce_mean(states_km_Z34[:, -28:], axis=1)
    mean_actions_km_Z34 = tf.reduce_mean(actions_km_Z34, axis=1)

    mean_states_km_Z5_T1_T2 = tf.reduce_mean(states_km_Z5_T1_T2[:, -28:], axis=1)
    mean_actions_km_Z5_T1_T2 = tf.reduce_mean(actions_km_Z5_T1_T2, axis=1)

    mean_states_km_sprinting = tf.reduce_mean(states_km_sprinting[:, -28:], axis=1)
    mean_actions_km_sprinting = tf.reduce_mean(actions_km_sprinting, axis=1)

    epsilon = 0.01

    # Calculate progress for each metric
    progress_total_km = mean_actions_total_km / (mean_states_total_km + epsilon)
    progress_km_Z34 = mean_actions_km_Z34 / (mean_states_km_Z34 + epsilon)
    progress_km_Z5_T1_T2 = mean_actions_km_Z5_T1_T2 / (mean_states_km_Z5_T1_T2 + epsilon)
    progress_km_sprinting = mean_actions_km_sprinting / (mean_states_km_sprinting + epsilon)
        
    median_running_progress =  np.median(progress_total_km.numpy())

    progress_reward_total_km = tf.nn.tanh(1.5 * progress_total_km)
    progress_km_sprinting = tf.nn.tanh(1.5 * progress_km_sprinting) * 0.1
    progress_km_Z34 = tf.nn.tanh(1.5 * progress_km_Z34) * 0.2
    progress_km_Z5_T1_T2 = tf.nn.tanh(1.5 * progress_km_Z5_T1_T2) * 0.2

    progress_total_reward = progress_reward_total_km + progress_km_sprinting + progress_km_Z34 + progress_km_Z5_T1_T2

    return  progress_total_reward, median_running_progress

def get_injury_score(self, states, actions):
    last_14_days = prepare_data_subjective_parameter_forecaster(states=states, actions=actions, min_max_values=self.min_max_values, standardised_min_max_values=self.standardised_min_max_values)
    subjective_params = self.predict_subjective_parameters(last_14_days)['output_0']
    last_7_days = prepare_data_injury_model(last_14_days=last_14_days, subjective_params=subjective_params)
    injury_scores = self.predict_injury_risk(last_7_days)['output_0']

    return injury_scores

def get_hard_penalty(states, actions, min_max_values, epoch):

    states = InputData(states)
    actions = OutputData(actions)

    states_total_km = get_absolute_values(states['total km'], min_max_values,'total km')
    actions_total_km = get_absolute_values(actions['total km'], min_max_values,'total km')

    condition = test_overall_load(states_total_km, actions_total_km)

    compliance_ratio = tf.round((tf.reduce_sum(tf.cast(tf.equal(condition, 0), tf.float32)) / tf.size(condition, out_type=tf.float32)) * 100 * 100) / 100

    HARD_PENALTY =  0.2 * epoch

    penalties = condition * HARD_PENALTY
   
    return penalties, compliance_ratio