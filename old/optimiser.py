import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import load_model
from RunningDataset.RunningDataset import Preprocessor
import os 
import xgboost as xgb
import pickle

class ContextualBandit:
    """
    This class implements a contextual bandit with a neural network to predict the best action 
    given a state and an ε-greedy strategy for exploration.
    """
    def __init__(self, state_shape, action_shape, env):
        """
        Initializes the contextual bandit.

        Parameters:
        - state_shape: The dimension of the input feature vector (state).
        - action_shape: The dimension of the output vector (action).
        - env: Instance of enviroment to interact with during gradient descent.
        """
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.env = env
        self.model = self.create_model()

    def create_model(self):
        """
        Creates the neural network model which predicts actions based on state input.

        The network consists of two LSTM layers and one output layer that predicts
        continuous action values corresponding to the action size.
        """
        model = models.Sequential([
            layers.LSTM(64, input_shape=self.state_shape, return_sequences=False),
            layers.RepeatVector(self.action_shape[0]),
            layers.LSTM(64, return_sequences=True),
            layers.TimeDistributed(layers.Dense(self.action_shape[1]))
        ])
        self.optimizer = optimizers.Adam(learning_rate=0.01)
        return model

    def get_actions(self, states):
        """
        Decides an action based on the current state 
        At a later stage an approach for exploring will be implemented - currently the model is only 
        exploiting (= train weights based on current position on loss function.
        """
        return self.model(states, training=True)

    def train(self, states):
        """
        Trains the model using the state, action taken, and reward received.
        The loss is calculated as the negative reward.
        """
        with tf.GradientTape() as tape:
            actions = self.get_actions(states)
            rewards = self.env.get_rewards(actions, states)
            loss = self.compute_loss(states, actions, rewards)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # Check gradients
        if any(grad is None for grad in gradients):
            raise ValueError("At least one gradient is None.")
        elif all(tf.reduce_sum(tf.abs(grad)) == 0 for grad in gradients):
            raise ValueError("All gradients are zero.")
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return -loss

    def compute_loss(self, predicted_actions, taken_actions, rewards):
        """
        Computes the loss function used to train the model.
        This is defined as the average negative reward
        """
        total_loss = -tf.reduce_mean(rewards)
        return total_loss

class Environment:
    """
    A simple environment that provides states and calculates rewards.
    """
    def __init__(self, data):
        """
        Initializes the environment with a dataset of states.
        """
        self.data = data
        #self.subjective_parameter_forecaster = tf.saved_model.load('final/subjective_parameter')
        #self.predict_subjective_parameters = self.subjective_parameter_forecaster.signatures['serving_default']
        #self.injury_model = tf.saved_model.load('final/injury')
        #self.predict_injury_risk = self.injury_model.signatures['serving_default']
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

    def get_states(self, indices):
        """
        Retrieves a batch of states from the dataset at a specified rage index.
        """
        return self.data[indices]

    def get_rewards(self, actions, states):
        """
        Calculates the reward by combining outputs from progress and injury functions, and any penalties.
        """
        progress = self.get_progress(states, actions)
        injury_scores = self.get_injury_score(states, actions)
        hard_penalties = self.get_hard_penalty(states, actions)
        rewards = progress - injury_scores - hard_penalties
        return rewards
    
    def get_progress(self, states, actions):
        return tf.reduce_mean(actions, axis=(1, 2)) - tf.reduce_mean(states, axis=(1, 2))

    def get_injury_score(self, states, actions):
        return tf.zeros(len(actions))
    
    def get_hard_penalty(self, states, actions):
        return tf.zeros(len(actions))

def train_optimiser(): 
    preprocessor = Preprocessor()
    X_train = preprocessor.preprocess()
    env = Environment(X_train)
    bandit = ContextualBandit(state_shape=(21,10), action_shape=(7,7), env=env)
    epochs = 50
    batch_size = 500

    for epoch in range(epochs):
        for index in range(0, len(X_train), batch_size):
            state_batch = env.get_states(slice(index, index +batch_size))
            reward = bandit.train(state_batch)
            print(f"Epoch {epoch + 1}, Batch starting at {index + 1}, Average reward: {reward}")

if __name__ == "__main__":
    train_optimiser()