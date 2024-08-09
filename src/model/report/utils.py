import tensorflow as tf
import numpy as np
from datetime import datetime
import json
import numpy as np
import os
import sys

# Set up the path to import get_absolute_values
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
src_path = os.path.abspath(os.path.join(current_dir, '../../'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from model.environment.reward_functions.utils.utils import get_absolute_values
from training_data.training_data import InputData, OutputData

def add_to_report(report, epoch_number, states, actions, rewards, athlete_ids_batch):
    # Append the current batch data to the report
    if tf.is_tensor(states):
        states = states.numpy()
    if tf.is_tensor(actions):
        actions = actions.numpy()
    if tf.is_tensor(rewards):
        rewards = rewards.numpy()

    report[epoch_number]['states'].append(states)
    report[epoch_number]['actions'].append(actions)
    report[epoch_number]['rewards'].append(rewards)
    report[epoch_number]['athlete_ids'].append(athlete_ids_batch)


def save_report(report, filepath, timestamp):
    json_report = {}
    for epoch, data in report.items():
        json_report[f'epoch_{epoch}'] = {
            'states': {idx: state for idx, state in enumerate(np.concatenate(data['states'], axis=0).tolist())},
            'actions': {idx: action for idx, action in enumerate(np.concatenate(data['actions'], axis=0).tolist())},
            'rewards': np.concatenate(data['rewards'], axis=0).tolist(),
            'athlete_ids': np.concatenate(data['athlete_ids'], axis=0).tolist()
        }

    with open(filepath + 'run_' + timestamp + '.json', 'w') as json_file:
        json.dump(json_report, json_file, indent=4)
    
    print("Report for run at timestamp ", timestamp, " saved successfully") 

def compute_metrics(report_filepath, min_max_filepath):
    with open(min_max_filepath, 'r') as file:
        min_max_values = json.load(file)

    metrics_dict = {}
    variables = [
        'nr. sessions', 'total km', 'km Z3-4', 'km Z5-T1-T2',
        'km sprinting', 'strength training', 'hours alternative'
    ]

    with open(report_filepath, 'r') as json_file:
        report = json.load(json_file)

    for epoch, data in report.items():
        states = np.array(list(data['states'].values()))
        actions = np.array(list(data['actions'].values()))

        # Convert states and actions using InputData and OutputData
        states = InputData(states)
        actions = OutputData(actions)

        states_total = {var: get_absolute_values(states[var], min_max_values, var) for var in variables}
        actions_total = {var: get_absolute_values(actions[var], min_max_values, var) for var in variables}

        metrics_dict[epoch] = {}
        
        for var in variables:
            state_values = states_total[var]
            action_values = actions_total[var]

            # Compute mean values for the last 28 days
            mean_states = np.mean(state_values, axis=1)
            mean_actions = np.mean(action_values, axis=1)
            
            # Compute increase of mean actions vs. states
            increase = np.divide(mean_actions, mean_states, out=np.zeros_like(mean_actions), where=mean_states != 0)
            
            # Flatten the arrays
            state_values_flat = state_values.flatten()
            action_values_flat = action_values.flatten()

            # Create the dictionary for mean_states and mean_actions
            mean_states_dict = {idx: value for idx, value in enumerate(mean_states)}
            mean_actions_dict = {idx: value for idx, value in enumerate(mean_actions)}

            # Create the dictionary for state_values_flat and action_values_flat
            state_values_flat_dict = {f"{idx % 7}.{idx // 7}": value for idx, value in enumerate(state_values_flat)}
            action_values_flat_dict = {f"{idx % 7}.{idx // 7}": value for idx, value in enumerate(action_values_flat)}

            metrics_dict[epoch][var] = {
                'mean_states': mean_states_dict,
                'mean_actions': mean_actions_dict,
                'increase': increase.tolist(),
                'state_values_flat': state_values_flat_dict,
                'action_values_flat': action_values_flat_dict
            }
    
    return metrics_dict, variables