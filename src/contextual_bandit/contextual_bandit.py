import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import load_model

class OutputLayer(layers.Layer):
    def __init__(self, action_shape, beta_means=0.1, beta_log_vars=0.1):
        super(OutputLayer, self).__init__()
        self.action_shape = action_shape
        self.beta_means = beta_means
        self.beta_log_vars = beta_log_vars

    def stretched_sigmoid(self, x, beta=0.2):
        """Apply sigmoid function with a stretch factor beta to input x."""
        return tf.sigmoid(x * beta)
    
    def scale_and_shift(self, x, min_val, max_val):
        """Scales and shifts sigmoid outputs to a specified range [min_val, max_val]."""
        return min_val + (max_val - min_val) * x

    def call(self, inputs):
        """
        Assumes input is already shaped correctly as [batch, height, width, channels]
        where channels = 2 for means and log variances.
        """
        means = inputs[..., 0]  # Extracting means
        log_vars = inputs[..., 1]  # Extracting log variances

        # Apply the stretched sigmoid and scale for means and log variances
        means = self.stretched_sigmoid(means, self.beta_means)
        #means = self.scale_and_shift(means, 0, 1)

        log_vars = self.stretched_sigmoid(log_vars, self.beta_log_vars)
        #log_vars = self.scale_and_shift(log_vars, -4.605, 0.811)

        # Reassembling the outputs into the correct format
        return tf.stack([means, log_vars], axis=-1)

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
        model = tf.keras.Sequential([
            layers.Input(shape=self.state_shape),  # Assuming an input sequence of 14 timesteps
            layers.LSTM(64, return_sequences=True),  # First LSTM layer
            layers.BatchNormalization(),
            layers.LSTM(64, return_sequences=True),  # First LSTM layer
            layers.BatchNormalization(),
            layers.LSTM(7, return_sequences=True),  # Adjust this LSTM to output 7 timesteps with 7 features each
            layers.TimeDistributed(layers.Dense(7)),  # E
            # layers.Dense(np.prod(self.action_shape)), #,2))
            # layers.Reshape((self.action_shape[0], self.action_shape[1]))#, 2))
            #OutputLayer(self.action_shape)
        ])
        self.optimizer = optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer=self.optimizer)
        return model

    def get_actions(self, states, training=False):
        """
        Generates actions based on model predictions using Gaussian distributions.
        During training, it samples actions to explore the action space effectively,
        and calculates log probabilities needed for gradient estimation in policy gradients.

        Params:
            states (Tensor): The current states from the environment, input to the model.
            training (bool): Whether the model is in training mode (True) or inference mode (False).

        Returns:
            Tensor: Actions as a tensor of shape (batch_size, 7, 7), sampled or deterministic.
            Tensor Log probabilities of sampled actions, returned only during training. These log probabilities are used to:
                - Calculate gradients for updating the model: In policy gradient methods, the gradient of the expected reward is estimated as the gradient of the log probability of the taken action scaled by the received reward.
                - Encourage exploration: By sampling actions based on their probabilities, the model explores a variety of actions, learning about their consequences, which enriches the training process and helps avoid local optima.
        """
       
        # predictions = self.model(states)  # Expected shape: (batch_size, 7, 7, 2)
        # means = predictions[:, :, :, 0]  # Mean, shape (batch_size, 7, 7)
        # print(f"means: {tf.reduce_mean(means)}")
        # log_vars = predictions[:, :, :, 1]  # Log variance, shape (batch_size, 7, 7)
        # print(f"logvars: {tf.reduce_mean(log_vars)}")
        # std_devs = tf.exp(log_vars / 2.0)  # Standard deviations, shape (batch_size, 7, 7)
        # print(f"stdv: {tf.reduce_mean(std_devs)}")

        # if training:
        #     # Sample actions based on the predicted Gaussian parameters
        #     normal_dist = tf.random.normal(shape=means.shape)
        #     actions = means + normal_dist * std_devs
        #     # Calculate log probabilities for the sampled actions
        #     log_probs = -0.5 * tf.reduce_sum(((actions - means) ** 2) / (std_devs ** 2) + log_vars + tf.math.log(2.0 * np.pi), axis=[1, 2])
        #     #log_probs = tf.clip_by_value(log_probs, -4000, 4000)
        #     return actions, log_probs
        # else:
        #     # Inference mode: use the means as deterministic actions
        #     actions = means
        #     return actions
        return self.model(states, training = training)

    def train(self, states):
        """
        Trains the model using states from the environment and corresponding rewards.
        Utilizes policy gradients to update model parameters, accommodating non-differentiable
        components of the reward through sampling and log probabilities.

        Args:
            states (Tensor): Current states from the environment to feed the model.

        Returns:
            float: Negative of the average loss (reward) during training.
        """
        with tf.GradientTape() as tape:
            actions = self.get_actions(states, training = True) # log_probs
            rewards = self.env.get_rewards(actions, states)
            loss = -tf.reduce_mean(rewards)  # Policy gradient update #log_probs * rewards
            #print(f"log_probs: {tf.reduce_mean(log_probs)}")
            print(f"rewards: {tf.reduce_mean(rewards)}")
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # Check gradients
        if any(grad is None for grad in gradients):
            raise ValueError("At least one gradient is None.")
        elif all(tf.reduce_sum(tf.abs(grad)) == 0 for grad in gradients):
            raise ValueError("All gradients are zero.")
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return -loss