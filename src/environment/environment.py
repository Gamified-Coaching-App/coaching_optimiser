import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import load_model
import os 
import xgboost as xgb
import pickle
import os 
from environment.reward_functions.reward_functions import get_progress, get_injury_score, get_hard_penalty

class Environment:
    """
    A simple environment that provides states and calculates rewards.
    """
    def __init__(self, data):
        """
        Initializes the environment with a dataset of states.
        """
        self.data = data
        self.subjective_parameter_forecaster = tf.saved_model.load('../../coaching_subjective_parameter_forecast/model/subjective_parameter')
        self.predict_subjective_parameters = self.subjective_parameter_forecaster.signatures['serving_default']
        self.injury_model = tf.saved_model.load('../../coaching_injury_prediction_model/exploration/LSTM/model/injury_prediction')
        self.predict_injury_risk = self.injury_model.signatures['serving_default']
        #self.injury_models = self.load_injury_models()
        print("Environment initialized")

    def load_injury_models(self):
        """
        Loads all XGBoost models from pickle files in the specified directory.
        """
        directory = '../../coaching_injury_prediction_model/models'
        models = []
        for filename in os.listdir(directory):
            if filename.endswith('.pkl'):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'rb') as file:
                    model = pickle.load(file)
                    models.append(model)
        return models

    def get_states(self, indices):
        """
        Retrieves a batch of states from the dataset at a specified rage index.
        """
        return self.data[indices]

    def get_rewards(self, actions, states):
        """
        Calculates the reward by combining outputs from progress and injury functions, and any penalties.
        """
        progress = get_progress(states, actions)
        injury_scores = get_injury_score(self, states, actions)
        hard_penalties = get_hard_penalty(states, actions)
        rewards = progress - injury_scores - hard_penalties 
        return rewards