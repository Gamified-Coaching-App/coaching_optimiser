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

def add_to_grid_search_report(report_dict, model_config, epoch, loss):
    arch_key = str(model_config)
    if arch_key not in report_dict:
        report_dict[arch_key] = {'losses': {}}
    report_dict[arch_key]['losses'][str(epoch)] = round(float(loss),4)

def train_optimiser(): 
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    configure_gpu()
    tf.config.set_visible_devices([], 'GPU')
    with h5py.File('./data/processed_data.h5', 'r') as hf:
        X_train = hf['X_train'][:]
        X_test = hf['X_test'][:]
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    athlete_ids_train = np.genfromtxt('data/athlete_ids_train.csv', delimiter=',', dtype=int)
    athlete_ids_test = np.genfromtxt('data/athlete_ids_test.csv', delimiter=',', dtype=int)
    
    env = Environment(X_train)
    epochs = 1#500
    batch_size = 4324#200#4243
    training_report = defaultdict(lambda: defaultdict(list))
    architecture_report = {}

    for model_config in model_configs:
        bandit = ContextualBandit(state_shape=(56,10), action_shape=(7,7), env=env, config=model_config)
        print(f"Training model with config:")
        bandit.model.summary()
        for epoch in range(epochs):
            all_gradients_zero_all_batches = []
            epoch_reward_sum = 0
            for index in range(0, len(X_train), batch_size):
                state_batch = env.get_states(slice(index, index + batch_size))
                athlete_id_batch = athlete_ids_train[index: index + batch_size]
                avg_reward, actions, rewards, all_gradients_zero = bandit.train(state_batch, athlete_id_batch, epoch)
                all_gradients_zero_all_batches.append(all_gradients_zero)
                if (epoch)% 10 == 0:
                    add_to_report(training_report, epoch, state_batch, actions, rewards, athlete_id_batch)
                epoch_reward_sum += avg_reward

            epoch_reward = epoch_reward_sum / int((len(X_train) / batch_size))
            print(f"Epoch {epoch + 1:6}, epoch reward: {round(float(epoch_reward), 1):15.1f}")
            if epoch % 10 == 0:
                add_to_grid_search_report(architecture_report, model_config, epoch + 1, float(epoch_reward))
            if all(all_gradients_zero_all_batches):
                print(f"Base training finished at epoch {epoch} with loss {round(float(epoch_reward),3)}.")
                add_to_grid_search_report(architecture_report, model_config, epoch + 1, float(epoch_reward))
                break
    
        test_loss = bandit.test(X_test)
        print(f"Test loss: {round(float(test_loss),4)}")
        add_to_grid_search_report(architecture_report, model_config, epoch='test', loss=float(test_loss))

    with open('report.json', 'w') as report_file:
        json.dump(architecture_report, report_file, indent=1)
    
    bandit.model.export('../model_export/export')
    #save_report(training_report, 'report/reports_data/', timestamp)
    model_path = '../model_export/export'  # Specify the correct path to your model
    model = tf.saved_model.load(model_path)

    # Assume the model uses the 'serving_default' signature
    predict_fn = model.signatures['serving_default']

    # Prepare input data
    model_path = '../model_export/export'  # Adjust the path as needed
    model = tf.saved_model.load(model_path)

    # Assume the model uses the 'serving_default' signature
    predict_fn = model.signatures['serving_default']

    # Prepare input data
    # Example: Batch of 1, shape (1, 56, 10)
    input_data = np.ones((1, 56, 10), dtype=np.float32)

    # Convert the input data to a TensorFlow tensor
    input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

    # Run inference on the model
    output = predict_fn(input_tensor)

    # Access the output by key (assuming 'output_0' is the key for the model output)
    # Replace 'output_0' with the actual key name if it's different
    output_tensor = output['output_0'].numpy()

    # Print the output
    print("Model output for input_data:", output_tensor.shape)


if __name__ == "__main__":
    train_optimiser()