import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import load_model
import os 
import xgboost as xgb
import pickle
import os 
from environment.reward_functions.reward_functions import get_progress, get_injury_score, get_hard_penalty
import json
from scipy.stats import pearsonr, spearmanr

"""
Environment class that provides the ContextualBandit with states and calculates rewards
"""
class Environment:
    def __init__(self, data):
        """
        __init__ initialises the environment with a dataset of states, loads models to make predictions and loads min-max values
        """
        self.data = data
        self.subjective_parameter_forecaster = tf.saved_model.load('../../../coaching_subjective_parameter_forecast/model/subjective_parameter_forecaster')
        self.predict_subjective_parameters = self.subjective_parameter_forecaster.signatures['serving_default']
        self.injury_predictor = tf.saved_model.load('../../../coaching_injury_prediction_model/surrogate_model/final')
        self.predict_injury_risk = self.injury_predictor.signatures['serving_default']
        with open('data/min_max_values.json', 'r') as file:
            self.min_max_values = json.load(file)
        with open('data/standardised_min_max_values.json', 'r') as file:
            self.standardised_min_max_values = json.load(file)
        print("Environment initialized")

    def get_states(self, indices):
        """
        function retrieves a batch of states from the dataset at a specified rage index
        """
        return self.data[indices]

    def get_rewards(self, actions, states, epoch):
        """
        function to compute rewards based on progress, injury risk and penalties
        """
        progress, median_running_progress = get_progress(states, actions, self.min_max_values)
        injury_scores = get_injury_score(self, states, actions)
        hard_penalties, compliance_ratio = get_hard_penalty(states, actions, self.min_max_values, epoch)
        rewards = progress  - injury_scores - hard_penalties
        return rewards, median_running_progress, compliance_ratio