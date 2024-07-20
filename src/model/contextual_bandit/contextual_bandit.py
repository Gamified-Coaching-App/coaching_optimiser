import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import load_model

class ContextualBandit(tf.Module):
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
        - env: Instance of environment to interact with during gradient descent.
        """
        super(ContextualBandit, self).__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.env = env
        self.model = self.create_model()
        self.optimizer = optimizers.Adam(learning_rate=0.01)

    def create_model(self):
        """
        Creates the neural network model which predicts actions based on state input.

        The network consists of two LSTM layers and one output layer that predicts
        continuous action values corresponding to the action size.
        """
        model = tf.keras.Sequential([
            layers.Input(shape=self.state_shape),
            layers.LSTM(64, return_sequences=True),
            layers.BatchNormalization(),
            layers.LSTM(64, return_sequences=True),
            layers.BatchNormalization(),
            layers.LSTM(7, return_sequences=True),
            layers.TimeDistributed(layers.Dense(7))
        ])
        return model

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 21, 10], dtype=tf.float32)])
    def serve(self, inputs):
        return self.model(inputs)

    def get_actions(self, states, training=False):
        """
        Generates actions based on model predictions.
        """
        return self.model(states, training=training)

    def train(self, states, athlete_ids):
        """
        Trains the model using states from the environment and corresponding rewards.
        """
        with tf.GradientTape() as tape:
            actions = self.get_actions(states, training=True)
            rewards = self.env.get_rewards(actions, states)
            loss = -tf.reduce_mean(rewards)
            print(f"rewards: {tf.reduce_mean(rewards)}")
        gradients = tape.gradient(loss, self.model.trainable_variables)
        if any(grad is None for grad in gradients):
            raise ValueError("At least one gradient is None.")
        elif all(tf.reduce_sum(tf.abs(grad)) == 0 for grad in gradients):
            raise ValueError("All gradients are zero.")
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return -loss, actions, rewards