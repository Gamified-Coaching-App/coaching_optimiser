import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import load_model
from preprocessor import Preprocessor
import os 
import xgboost as xgb
import pickle


class Environment:
    """
    A simple environment that provides states and calculates rewards.
    """
    def __init__(self, data):
        """
        Initializes the environment with a dataset of states.
        """
        self.data = data
        self.subjective_parameter_forecaster = tf.saved_model.load('final/subjective_parameter')
        self.predict_subjective_parameters = self.subjective_parameter_forecaster.signatures['serving_default']
        self.injury_model = tf.saved_model.load('final/injury')
        self.predict_injury_risk = self.injury_model.signatures['serving_default']
        print("Environment initialized")

    def load_injury_models(self):
        """
        Loads all XGBoost models from pickle files in the specified directory.
        """
        directory = '../../coaching_injury_prediction_model/model'
        models = []
        for filename in os.listdir(directory):
            if filename.endswith('.pkl'):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'rb') as file:
                    model = pickle.load(file)
                    models.append(model)
        return models

    def get_state(self, index):
        """
        Retrieves a state from the dataset at a specified index.
        """
        return self.data[index]

    def get_reward(self, action, state):
        """
        Calculates the reward by combining outputs from progress and injury functions, and any penalties.
        """
        progress = self.get_progress(state, action)
        injury_score = self.get_injury_score(state, action)
        hard_penalty = self.get_hard_penalty(state, action)
        reward = progress - injury_score - hard_penalty
        return reward
    
    def get_progress(self, state, action):
        """
        Placeholder for calculating progress based on the state and action.

        Implement this function based on how progress should be quantified in your context.
        """
        index_total_km = 1
        index_km_z3_4 = 2
        index_km_z5_t1_t2 = 3
        index_km_sprinting = 4
        index_strength_training = 5
        index_hours_alternative = 6

        # Mean KM
        km_total_future = np.mean(action[:,index_total_km])
        km_total_past = np.mean(state[:, index_total_km])
        # Mean Zone mid
        km_mid_future = np.mean(action[:,index_km_z3_4])
        km_mid_past = np.mean(state[:, index_km_z3_4])
        # Mean Zone high
        km_high_future= np.mean(action[:, index_km_z5_t1_t2])
        km_high_past = np.mean(state[:, index_km_z5_t1_t2])
        # Mean Zone sprint
        km_sprint_future = np.mean(action[:, index_km_sprinting])
        km_sprint_past = np.mean(state[:, index_km_sprinting])
        # Mean strength training
        km_strength_future = np.mean(action[:, index_strength_training])
        km_strength_past = np.mean(state[:, index_strength_training])
        # Mean alternative training
        km_alt_future = np.mean(action[:, index_hours_alternative])
        km_alt_past = np.mean(state[:, index_hours_alternative])

        scaling_factor = 1
        #
        factor_total_km = 1
        factor_km_z3_4 = 0.2
        factor_z5_t1_t2 = 0.5
        factor_sprinting = 0.3
        factor_strength = 0.1
        factor_alt = 0.1

        progress = factor_total_km * (km_total_future - km_total_past) +\
                    factor_km_z3_4 * (km_mid_future - km_mid_past) +\
                    factor_z5_t1_t2 * (km_high_future - km_high_past) +\
                    factor_sprinting * (km_sprint_future - km_sprint_past) +\
                    factor_strength * (km_strength_future - km_strength_past) +\
                    factor_alt * (km_alt_future - km_alt_past)
        
        return progress * scaling_factor

    def get_injury_score(self, state, action):
        """
        Placeholder for calculating an injury score based on the state and action.
        - 'action' is expected to be a 7x7 array.
        - 'state' is a 21x10 array.
        Implement this function based on your injury model's prediction mechanism.
        """
        # Extend 'action' to 7x10 by appending a 7x3 zero matrix to it
        full_data = np.concatenate((action, np.zeros((7, 3))), axis=1)
        # Extract the last 14 rows from the 'state' which is a 21x10 array
        last_fourteen = state[-14:]  # This slices the last 14 rows
        # Concatenate full_data (7x10) with last_fourteen (14x10) vertically
        concatenated_data = np.concatenate((full_data, last_fourteen), axis=0)
        concatenated_data = np.expand_dims(concatenated_data, axis=0)

        # Using the correct signature to make predictions
        predict = self.subjective_parameter_forecaster.signatures['serving_default']
        prediction = self.predict_subjective_parameters(tf.convert_to_tensor(concatenated_data, dtype=tf.float32))['output_0']
        prediction = prediction.numpy()
        concatenated_data[0, -7:, -3:] = prediction[0, -7:, -3:]

        injury_data = concatenated_data[:,-7:,:]

        #print(self.subjective_parameter_forecaster.signatures)
        predict = self.injury_model.signatures['serving_default']
        prediction = predict(tf.convert_to_tensor(injury_data, dtype=tf.float32))['output_0']
        injury_score = prediction.numpy()[0, 0]

        return injury_score  
    
    def get_hard_penalty(self, state, action):
        return 0