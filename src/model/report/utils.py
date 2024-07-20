import tensorflow as tf
import h5py
import numpy as np
from datetime import datetime

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
    with h5py.File(filepath + 'run_' + timestamp + '.h5', 'w') as hf:
        for epoch, data in report.items():
            epoch_group = hf.create_group(f'epoch_{epoch}')
            
            epoch_group.create_dataset('states', data=np.concatenate(data['states'], axis=0))
            epoch_group.create_dataset('actions', data=np.concatenate(data['actions'], axis=0))
            epoch_group.create_dataset('rewards', data=np.concatenate(data['rewards'], axis=0))
            epoch_group.create_dataset('athlete_ids', data=np.concatenate(data['athlete_ids'], axis=0))
    print("Report for run at timestamp ", timestamp, " saved successfully")
