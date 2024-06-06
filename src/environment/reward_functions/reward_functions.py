import tensorflow as tf
import pandas as pd
from environment.reward_functions.utils.utils import *

def get_progress(states, actions):

    states_total_km = get_variable(states, 'km total')
    actions_total_km = get_variable(actions, 'km total')
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

def get_hard_penalty(states, actions):
    HARD_PENALTY = 10000.0

    states_total_km = get_variable(states, 'km total')
    actions_total_km = get_variable(actions, 'km total')
    avg_states_km_total = tf.reduce_mean(states_total_km, axis=1)
    avg_actions_km_total = tf.reduce_mean(actions_total_km, axis=1)

    condition = avg_actions_km_total > 2 * avg_states_km_total

    penalties = tf.where(condition, 
                         tf.fill(dims=tf.shape(condition), value=float(HARD_PENALTY)),  
                         tf.zeros_like(condition, dtype=tf.float32))
    
    return penalties
    
