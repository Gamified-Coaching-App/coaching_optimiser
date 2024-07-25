from environment.environment import Environment
from contextual_bandit.contextual_bandit import ContextualBandit
from report.utils import add_to_report, save_report
from collections import defaultdict
import tensorflow as tf
import h5py
import numpy as np
from datetime import datetime

def train_optimiser(): 
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    with h5py.File('./data/processed_data.h5', 'r') as hf:
        X_train = hf['X_train'][:]
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    athlete_ids = np.genfromtxt('data/athlete_ids.csv', delimiter=',', dtype=int)
    
    env = Environment(X_train)
    bandit = ContextualBandit(state_shape=(56,10), action_shape=(7,7), env=env)
    epochs = 1
    batch_size = 1000
    training_report = defaultdict(lambda: defaultdict(list))

    for epoch in range(epochs):
        for index in range(0, len(X_train), batch_size):
            state_batch = env.get_states(slice(index, index + batch_size))
            athlete_id_batch = athlete_ids[index: index + batch_size]
            avg_reward, actions, rewards = bandit.train(state_batch, athlete_id_batch)
            add_to_report(training_report, epoch, state_batch, actions, rewards, athlete_id_batch)
            print(f"Epoch {epoch + 1}, Batch starting at {index + 1}, Average reward: {avg_reward}")
    
    bandit.model.export('../model_export/export')
    save_report(training_report, 'report/reports_data/', timestamp)

if __name__ == "__main__":
    train_optimiser()