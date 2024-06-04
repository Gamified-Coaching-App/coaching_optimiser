import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import load_model
from preprocessor import Preprocessor
import os 
import xgboost as xgb
import pickle

class ContextualBandit:
    """
    This class implements a contextual bandit with a neural network to predict the best action 
    given a state and an ε-greedy strategy for exploration.
    """
    def __init__(self, state_shape, action_shape, epsilon=0.0):
        """
        Initializes the contextual bandit.

        Parameters:
        - state_shape: The dimension of the input feature vector (state).
        - action_shape: The dimension of the output vector (action).
        - epsilon: The probability of choosing a random action (exploration factor).
        """
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.epsilon = epsilon
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
        Decides an action based on the current state using an ε-greedy approach.

        With probability ε, a random action within the defined range is chosen to promote exploration.
        Otherwise, the model's prediction is used to exploit the learned policy.
        """
        if np.random.rand() < self.epsilon:
            # Exploration: Select random actions for each state in the batch
            random_actions = np.random.uniform(-1, 1, (len(states),) + self.action_shape)
            return tf.convert_to_tensor(random_actions, dtype=tf.float32), np.ones(len(states), dtype=bool)
        else:
            # Exploitation: Select the best actions predicted by the model for the batch
            return self.model(states, training=False), np.zeros(len(states), dtype=bool)

    def train(self, states, actions, rewards, exploring):
        """
        Trains the model using the state, action taken, and reward received.

        The loss is calculated as the negative reward scaled by the mean squared error
        between the predicted action and the action taken. This encourages the model
        to predict actions that lead to higher rewards.
        """
        # Skip training for the entire batch if all actions were chosen randomly
        if exploring.all():
            return
        
        valid_indices = ~exploring
        valid_states = states[valid_indices]
        valid_actions = actions[valid_indices]
        valid_rewards = rewards[valid_indices]

        with tf.GradientTape() as tape:
            predicted_actions = self.model(valid_states, training=True)
            loss = self.compute_loss(predicted_actions, valid_actions, valid_rewards)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def compute_loss(self, predicted_actions, taken_actions, rewards):
        """
        Computes the loss function used to train the model.

        This is defined as the negative reward times the mean squared error between
        the taken and predicted actions. It incentivizes minimizing the error for actions
        that lead to higher rewards.
        """
        action_loss = tf.reduce_mean(tf.square(taken_actions - predicted_actions), axis=[1, 2])
        total_loss = -tf.reduce_mean(rewards * action_loss)
        return total_loss