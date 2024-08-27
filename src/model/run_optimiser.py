from environment.environment import Environment
from contextual_bandit.contextual_bandit import ContextualBandit
from report.utils import add_to_report, save_report
from collections import defaultdict
import tensorflow as tf
import h5py
import numpy as np
from datetime import datetime
from config import model_configs
import json
import math
from sklearn.model_selection import train_test_split, KFold
from datetime import datetime


def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs Available: {gpus}")
        except RuntimeError as e:
            print(e)
    else: 
        print("No GPUs available.")

def train_optimiser(mode): 
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    configure_gpu()
    tf.config.set_visible_devices([], 'GPU')
    with h5py.File('./data/processed_data.h5', 'r') as hf:
        X = hf['X'][:]
    
    epochs = 1000

    training_report = {}

    if mode == 'final_training':

        X_train, X_temp = train_test_split(X, test_size=0.3, random_state=10)
        X_test, X_val = train_test_split(X_temp, test_size=0.5, random_state=10)

        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

        for model_config in model_configs:
            train_median_progress_epoch = []
            train_compliance_ratios_epoch = []
            val_median_progress_epoch = []
            val_compliance_ratio_epoch = []

            batch_size = model_config['batch_size']
            
            env = Environment(X_train)
            bandit = ContextualBandit(state_shape=(56,10), action_shape=(7,7), env=env, config=model_config)

            for epoch in range(epochs):
                batch_count=0
                all_gradients_zero_all_batches = []
                epoch_reward_sum = 0
                median_running_progress_sum = 0
                compliance_ratio_sum = 0
                for index in range(0, len(X_train), batch_size):
                    state_batch = env.get_states(slice(index, index + batch_size))
                    avg_reward, all_gradients_zero, median_running_progress, compliance_ratio = bandit.train(state_batch, epoch)
                    all_gradients_zero_all_batches.append(all_gradients_zero)

                    epoch_reward_sum += avg_reward
                    median_running_progress_sum += median_running_progress
                    compliance_ratio_sum += compliance_ratio
                    batch_count += 1

                epoch_reward = epoch_reward_sum / batch_count
                median_running_progress = median_running_progress_sum / batch_count
                compliance_ratio = compliance_ratio_sum / batch_count
                train_median_progress_epoch.append(float(median_running_progress))
                train_compliance_ratios_epoch.append(float(compliance_ratio))
                print(f"Train set: Epoch {epoch + 1:6}, reward: {round(float(epoch_reward), 3):5.3f}, median runnign progress: {round(float(median_running_progress), 3):5.3f}, compliance ratio: {round(float(compliance_ratio), 3):5.3f}%")
                val_loss, val_median_running_progress, val_compliance_ratio = bandit.test(X_val, epoch)
                val_median_progress_epoch.append(float(val_median_running_progress))
                val_compliance_ratio_epoch.append(float(val_compliance_ratio))
                print(f"Validation set: Epoch {epoch + 1:6}, reward: {round(float(val_loss), 3):5.3f}, median runnign progress: {round(float(val_median_running_progress), 3):5.3f}, compliance ratio: {round(float(val_compliance_ratio), 3):5.3f}%")
                if all(all_gradients_zero_all_batches):
                    print(f"All gradients zero.")
                    break
                if float(val_compliance_ratio) > 99.9:
                    break

            test_loss, median_running_progress, compliance_ratio = bandit.test(X_test, epochs + 1)
            print(f"Test set loss: {round(float(test_loss),4)}, median running progress: {round(float(median_running_progress), 4)}, compliance ratio: {round(float(compliance_ratio), 0)}")
            training_report[model_config['name']] = {
                'train_median_progress': train_median_progress_epoch,
                'train_compliance_ratios': train_compliance_ratios_epoch,
                'val_median_progress': val_median_progress_epoch,
                'val_compliance_ratios': val_compliance_ratio_epoch,
                'test_median_progress': float(median_running_progress),
                'test_compliance_ratio': float(compliance_ratio)
            }
            bandit.model.export('../../model_export/' + model_config['name'])
            path = '../../model_export/' + model_config['name']
            model = tf.saved_model.load(path)
            predict = model.signatures['serving_default']
            inference = predict(X_test)['output_0'].numpy()
            with open('report/report_files/inference.json', 'w') as report_file:
                json.dump({
                    'X_test': X_test.numpy().tolist() if hasattr(X_test, 'numpy') else X_test.tolist(),
                    'inference': inference.tolist()
                }, report_file, indent=1)

        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        with open(f'report/report_files/training_report_{timestamp}.json', 'w') as report_file:
            json.dump(training_report, report_file, indent=1)

    elif mode == 'cross_validation':
        kf = KFold(n_splits=10, shuffle=True)
        for fold_index, (train_index, val_index) in enumerate(kf.split(X)):
            X_train, X_val = X[train_index], X[val_index]

            for model_config in model_configs:
                train_median_progress_epoch = []
                train_compliance_ratios_epoch = []
                val_median_progress_epoch = []
                val_compliance_ratio_epoch = []

                batch_size = model_config['batch_size']
                
                env = Environment(X_train)
                bandit = ContextualBandit(state_shape=(56,10), action_shape=(7,7), env=env, config=model_config)

                for epoch in range(epochs):
                    batch_count=0
                    all_gradients_zero_all_batches = []
                    epoch_reward_sum = 0
                    median_running_progress_sum = 0
                    compliance_ratio_sum = 0
                    for index in range(0, len(X_train), batch_size):
                        state_batch = env.get_states(slice(index, index + batch_size))
                        avg_reward, all_gradients_zero, median_running_progress, compliance_ratio = bandit.train(state_batch, epoch)
                        all_gradients_zero_all_batches.append(all_gradients_zero)

                        epoch_reward_sum += avg_reward
                        median_running_progress_sum += median_running_progress
                        compliance_ratio_sum += compliance_ratio
                        batch_count += 1

                    epoch_reward = epoch_reward_sum / batch_count
                    median_running_progress = median_running_progress_sum / batch_count
                    compliance_ratio = compliance_ratio_sum / batch_count
                    train_median_progress_epoch.append(float(median_running_progress))
                    train_compliance_ratios_epoch.append(float(compliance_ratio))
                    print(f"Train set: Epoch {epoch + 1:6}, reward: {round(float(epoch_reward), 3):5.3f}, median runnign progress: {round(float(median_running_progress), 3):5.3f}, compliance ratio: {round(float(compliance_ratio), 3):5.3f}%")
                    val_loss, val_median_running_progress, val_compliance_ratio = bandit.test(X_val, epoch)
                    val_median_progress_epoch.append(float(val_median_running_progress))
                    val_compliance_ratio_epoch.append(float(val_compliance_ratio))
                    print(f"Validation set: Epoch {epoch + 1:6}, reward: {round(float(val_loss), 3):5.3f}, median runnign progress: {round(float(val_median_running_progress), 3):5.3f}, compliance ratio: {round(float(val_compliance_ratio), 3):5.3f}%")
                    if all(all_gradients_zero_all_batches):
                        print(f"All gradients zero.")
                        break
                    if val_compliance_ratio > 99.9:
                        break

                training_report[f"{model_config['name']}_fold_{fold_index}"] = {
                    'val_median_progress': val_median_progress_epoch[-1],
                    'val_compliance_ratios': val_compliance_ratio_epoch[-1]
                }

            with open(f'report/report_files/cross_val_report.json', 'w') as report_file:
                json.dump(training_report, report_file, indent=1)
        
        else:
            raise ValueError("Invalid mode. Choose either 'final_training' or 'cross_validation'.")

if __name__ == "__main__":
    train_optimiser(mode='final_training')