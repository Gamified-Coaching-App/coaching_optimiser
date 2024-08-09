import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import load_model
from contextual_bandit.OutputLayer.OutputLayer import OutputLayer

class ContextualBandit(tf.Module):
    """
    This class implements a contextual bandit with a neural network to predict the best action 
    given a state and an ε-greedy strategy for exploration.
    """
    def __init__(self, state_shape, action_shape, env, config):
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
        self.model = self.create_model(config['layers'])
        if config['optimiser'] == 'adam':
            self.optimizer = optimizers.Adam(learning_rate=config['learning_rate'])
            print("Using Adam optimizer")
        else:
            self.optimizer = optimizers.Adadelta(learning_rate=config['learning_rate'])
            print("Using Adadelta optimizer")
        
    def create_model(self, config):
        """
        Creates the neural network model based on the given configuration.

        Args:
        config (dict): Configuration dictionary for the model.

        Returns:
        model (tf.keras.Model): Compiled neural network model.
        """
        original_input = layers.Input(shape=self.state_shape, name='original_input')
        x = original_input

        for layer_config in config:
            layer_type = layer_config['type']

            if layer_type == 'lstm':
                params = {
                    'units': layer_config['units'],
                    'return_sequences': layer_config['return_sequences'],
                }
                if 'dropout' in layer_config:
                    params['dropout'] = layer_config['dropout']
                if 'recurrent_dropout' in layer_config:
                    params['recurrent_dropout'] = layer_config['recurrent_dropout']
                if 'activation' in layer_config:
                    params['activation'] = layer_config['activation']
                x = layers.LSTM(**params)(x)

            elif layer_type == 'bidirectional_lstm':
                params = {
                    'units': layer_config['units'],
                    'return_sequences': layer_config['return_sequences'],
                }
                if 'dropout' in layer_config:
                    params['dropout'] = layer_config['dropout']
                if 'recurrent_dropout' in layer_config:
                    params['recurrent_dropout'] = layer_config['recurrent_dropout']
                if 'activation' in layer_config:
                    params['activation'] = layer_config['activation']
                x = layers.Bidirectional(layers.LSTM(**params))(x)

            elif layer_type == 'dense':
                params = {
                    'units': layer_config['units'],
                }
                if 'activation' in layer_config:
                    params['activation'] = layer_config['activation']
                if 'kernel_initializer' in layer_config:
                    params['kernel_initializer'] = layer_config['kernel_initializer']
                if 'bias_initializer' in layer_config:
                    params['bias_initializer'] = layer_config['bias_initializer']
                
                x = layers.Dense(**params)(x)

            elif layer_type == 'conv2d':
                x = layers.Conv2D(
                    filters=layer_config['filters'],
                    kernel_size=layer_config['kernel_size'],
                    activation=layer_config['activation'],
                    padding=layer_config.get('padding', 'valid')
                )(x)

            elif layer_type == 'max_pooling2d':
                x = layers.MaxPooling2D(pool_size=layer_config['pool_size'])(x)

            elif layer_type == 'batch_normalization':
                x = layers.BatchNormalization()(x)
            
            elif layer_type == 'conv1d':
                x = layers.Conv1D(
                    filters=layer_config['filters'],
                    kernel_size=layer_config['kernel_size'],
                    activation=layer_config['activation']
                )(x)

            elif layer_type == 'max_pooling1d':
                x = layers.MaxPooling1D(pool_size=layer_config['pool_size'])(x)

            elif layer_type == 'dropout':
                x = layers.Dropout(rate=layer_config['rate'])(x)

            elif layer_type == 'reshape':
                x = layers.Reshape(target_shape=layer_config['target_shape'])(x)
            
            elif layer_type == 'dropout':
                params = {
                    'rate': layer_config['rate'],
                }
                x = layers.Dropout(**params)(x)

            elif layer_type == 'alphadropout':
                params = {
                    'rate': layer_config['rate'],
                }
                x = layers.AlphaDropout(**params)(x)

            elif layer_type == 'activation':
                x = layers.Activation('relu')(x)

            elif layer_type == 'flatten':
                x = layers.Flatten()(x)

            elif layer_type == 'output':
                immediate_input = x
                output = OutputLayer()([immediate_input, original_input])
                model = models.Model(inputs=original_input, outputs=output)
                return model 

    def get_actions(self, states, training=False):
        """
        Generates actions based on model predictions.
        """
        return self.model(states, training=training)

    def train(self, states, athlete_ids, epoch):
        """
        Trains the model using states from the environment and corresponding rewards.
        """
        with tf.GradientTape() as tape:
            actions = self.get_actions(states, training=True)
            rewards = self.env.get_rewards(actions, states)
            loss = -tf.reduce_mean(rewards)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        all_gradients_zero = all(tf.reduce_sum(tf.abs(grad)).numpy() == 0 for grad in gradients)
        if any(grad is None for grad in gradients):
            raise ValueError("At least one gradient is None.")
        elif all_gradients_zero:
            print("All gradients are zero. Loss is ", loss.numpy())
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return -loss, actions, rewards, all_gradients_zero
    
    def test(self, test_states):
        """
        Tests the model using the given test set, makes inferences, and evaluates using the environment.
        
        Parameters:
        - test_states: The test set of input states.
        
        Returns:
        - rewards: The rewards obtained from the environment based on the model's actions.
        """
        actions = self.get_actions(test_states, training=False)
        rewards = self.env.get_rewards(actions, test_states)
        loss = -tf.reduce_mean(rewards)
        return -loss