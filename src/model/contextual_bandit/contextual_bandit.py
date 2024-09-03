import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import load_model
from keras_nlp.layers import TransformerEncoder, TransformerDecoder
from contextual_bandit.OutputLayer.OutputLayer import OutputLayer

"""
ContextualBandit class represents model learning optimisation
"""
class ContextualBandit(tf.Module):
    """
    __init__ initialises the ContextualBandit class by setting up model based on @config, connecting to environment @env
    """
    def __init__(self, state_shape, action_shape, env, config):
        super(ContextualBandit, self).__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.env = env
        self.model = self.create_model(config['layers'])
        if 'learning_rate_schedule' in config:
            lr = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=config['learning_rate_schedule']['initial_learning_rate'],
                decay_steps=config['learning_rate_schedule']['decay_steps'],
                warmup_target=config['learning_rate_schedule']['target_learning_rate'],
                warmup_steps=config['learning_rate_schedule']['warmup_steps'],
                alpha=config['learning_rate_schedule']['alpha'] 
            )
        else:
            lr = config['learning_rate']
        if config['optimiser'] == 'adam':
            self.optimizer = optimizers.Adam(learning_rate=lr, weight_decay=0.0)
            print("Using Adam optimizer")
        elif config['optimiser'] == 'adamw':
            self.optimizer = optimizers.AdamW(learning_rate=lr, weight_decay=0.0)
            print("Using AdamW optimizer with learning rate scheduling")
        else:
            self.optimizer = optimizers.Adadelta(learning_rate=config['learning_rate'])
            print("Using Adadelta optimizer")
        
    def create_model(self, config):
        """
        Creates the neural network model based on the given configuration.
        Returns compiled model
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

            elif layer_type == 'global_average_pooling1d':
                x = layers.GlobalAveragePooling1D()(x)

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

            elif layer_type == 'transformer_encoder':
                params = {
                    'num_heads': layer_config['num_heads'],
                    'intermediate_dim': layer_config['intermediate_dim'],
                    'dropout': layer_config.get('dropout'),
                    'activation': layer_config.get('activation'), 
                    'normalize_first': layer_config.get('normalize_first')
                }
                x = TransformerEncoder(**params)(x)
            
            elif layer_type == 'transformer_decoder':
                params = {
                    'num_heads': layer_config['num_heads'],
                    'intermediate_dim': layer_config['intermediate_dim'],
                    'dropout': layer_config.get('dropout'),
                    'activation': layer_config.get('activation'), 
                    'normalize_first': layer_config.get('normalize_first')
                }
                x = TransformerDecoder(**params)(x)

            elif layer_type == 'flatten':
                x = layers.Flatten()(x)

            elif layer_type == 'activation':
                output = layers.Activation(layer_config['activation'])(x)
                model = models.Model(inputs=original_input, outputs=output)
                return model

            elif layer_type == 'output':
                immediate_input = x
                output = OutputLayer()([immediate_input, original_input])
                model = models.Model(inputs=original_input, outputs=output)
                return model 

    def get_actions(self, states, training=False):
        """
        Generates actions based on model predictions
        """
        actions = self.model(states, training=training)
        if tf.reduce_any(tf.math.is_nan(actions)):
            if tf.reduce_any(tf.math.is_nan(states)):
                print("NaN values found in the 'states' tensor.")
            raise ValueError("NaN values found in the 'actions' tensor.")
        return actions

    def train(self, states, epoch):
        """
        Trains the model using states from the environment and computing the corresponding rewards
        """
        with tf.GradientTape() as tape:
            actions = self.get_actions(states, training=True)
            rewards, median_running_progress, compliance_ratio = self.env.get_rewards(actions, states, epoch)
            loss = -tf.reduce_mean(rewards)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        all_gradients_zero = all(tf.reduce_sum(tf.abs(grad)).numpy() == 0 for grad in gradients)
        if any(grad is None for grad in gradients):
            raise ValueError("At least one gradient is None.")
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return -loss, all_gradients_zero, median_running_progress, compliance_ratio
    
    def test(self, test_states, epoch):
        """
        Tests the model using the given test set, makes inferences, and evaluates using the environment.
        @epoch is equal to last training epoch + 1
        """
        actions = self.get_actions(test_states, training=False)
        rewards, median_running_progress, compliance_ratio = self.env.get_rewards(actions, test_states, epoch)
        loss = -tf.reduce_mean(rewards)
        return -loss, median_running_progress, compliance_ratio