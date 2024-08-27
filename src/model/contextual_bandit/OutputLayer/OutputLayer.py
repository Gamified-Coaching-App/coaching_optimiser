import tensorflow as tf
import tensorflow.keras.layers as layers
import os 
import sys
from contextual_bandit.OutputLayer.config import config
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
src_path = os.path.abspath(os.path.join(current_dir, '../../'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from training_data.training_data import InputData, OutputData
import json

class OutputLayer(layers.Layer):
    def __init__(self):
        super(OutputLayer, self).__init__()
        with open('data/min_max_values.json', 'r') as file:
            self.min_max_values = json.load(file)
        self.indexes = { 
            'total km': 0,
            'km Z3-4': 1,
            'km Z5-T1-T2': 2,
            'km sprinting': 3
        }
        self.lower_thresholds = {
            'nr. sessions': self.scale_absolute_value(config['lower_thresholds']['nr. sessions'], 'nr. sessions'),
            'total km': self.scale_absolute_value(config['lower_thresholds']['total km'], 'total km'),
            'km Z3-4': self.scale_absolute_value(config['lower_thresholds']['km Z3-4'], 'km Z3-4'),
            'km Z5-T1-T2': self.scale_absolute_value(config['lower_thresholds']['km Z5-T1-T2'], 'km Z5-T1-T2'),
            'km sprinting': self.scale_absolute_value(config['lower_thresholds']['km sprinting'], 'km sprinting'),
            'strength training': self.scale_absolute_value(config['lower_thresholds']['strength training'], 'strength training'),
            'hours alternative': self.scale_absolute_value(config['lower_thresholds']['hours alternative'], 'hours alternative')
        }
        self.upper_thresholds_ratios = {
            'km Z3-4': self.scale_ratio(config['upper_thresholds']['ratio_of_total_km']['km Z3-4'], 'km Z3-4'),
            'km Z5-T1-T2': self.scale_ratio(config['upper_thresholds']['ratio_of_total_km']['km Z5-T1-T2'], 'km Z5-T1-T2'),
            'km sprinting': self.scale_ratio(config['upper_thresholds']['ratio_of_total_km']['km sprinting'], 'km sprinting')
        }
        self.upper_thresholds_historic_comparison = {
            'total km': self.scale_absolute_value(config['upper_thresholds']['increase_vs_history']['total km'], 'total km'),
            'hours alternative': self.scale_absolute_value(config['upper_thresholds']['increase_vs_history']['hours alternative'], 'hours alternative'),
        }
        self.upper_thresholds_absolute = {
            'nr. sessions': self.scale_absolute_value(config['upper_thresholds']['absolute']['nr. sessions'], 'nr. sessions'),
            'strength training': self.scale_absolute_value(config['upper_thresholds']['absolute']['strength training'], 'strength training'),
        }
        self.days_for_historic_comparison = config['upper_thresholds']['days_for_historic_comparison']
        print("OutputLayer initialized with values:")
        print("Lower Thresholds:", self.lower_thresholds)
        print("Upper Thresholds Ratios:", self.upper_thresholds_ratios)
        print("Upper Thresholds Absolute Historic Comparison:", self.upper_thresholds_historic_comparison)
        print("Upper Thresholds Absolute:", self.upper_thresholds_absolute)

    def scale_absolute_value (self, x, variable):
        return ((float(x) - float(self.min_max_values[variable]['min'])) / (float(self.min_max_values[variable]['max']) - float(self.min_max_values[variable]['min'])))
    
    def scale_ratio(self, percentage, variable):
        range_total_km = self.min_max_values['total km']['max'] - self.min_max_values['total km']['min']
        range_variable = self.min_max_values[variable]['max'] - self.min_max_values[variable]['min']
        return percentage * range_variable / range_total_km

    def call(self, inputs):
        immediate_input, states = inputs
        states_indexable = InputData(states)
        
        tf.debugging.assert_equal(tf.shape(immediate_input)[1:], (7, 4), message="immediate_input shape is incorrect, got {} instead of (7,4)".format(tf.shape(immediate_input)))
        tf.debugging.assert_equal(tf.shape(states)[1:], (56, 10), message="original_input shape is incorrect, got {} instead of (56,10)".format(tf.shape(states)))
        
        output = tf.nn.relu(immediate_input)
        output = self.enforce_upper_threshold_absolute(output, states)
        output = self.enforce_lower_threshold(output)
        #output = self.enforce_rest_days(output, states_indexable)
        output = self.enforce_logical_correctness(output)
        output = tf.nn.relu(output)
        return output

    def build_lower_threshold_tensor(self, inputs):
        batch_size = tf.shape(inputs)[0]
        num_timesteps = tf.shape(inputs)[1]
        lower_threshold_values = [
            self.lower_thresholds['total km'],
            self.lower_thresholds['km Z3-4'],
            self.lower_thresholds['km Z5-T1-T2'],
            self.lower_thresholds['km sprinting']
        ]
        lower_threshold_tensor = tf.constant(lower_threshold_values, shape=(1, 1, len(lower_threshold_values)))
        lower_threshold_tensor = tf.tile(lower_threshold_tensor, [batch_size, num_timesteps, 1])
        
        tf.debugging.assert_equal(tf.shape(lower_threshold_tensor)[1:], (7, 4), message="immediate_input shape is incorrect, got {} instead of (7,4)".format(tf.shape(lower_threshold_tensor)))

        return lower_threshold_tensor
    
    def enforce_lower_threshold(self, input):
        lower_threshold_tensor = self.build_lower_threshold_tensor(input)
        return tf.where(input < lower_threshold_tensor, tf.nn.relu(-input), tf.nn.relu(input))
    
    def enforce_upper_threshold_absolute(self, actions, states):
        states_indexable = InputData(states)
        actions= self.enforce_upper_threshold_per_variable(actions, None, states_indexable, 'total km')

        actions_indexable = OutputData(actions)
        actions= self.enforce_upper_threshold_per_variable(actions, actions_indexable, None, 'km Z3-4')
        actions= self.enforce_upper_threshold_per_variable(actions, actions_indexable, None, 'km Z5-T1-T2')
        actions= self.enforce_upper_threshold_per_variable(actions, actions_indexable, None, 'km sprinting')
        
        return actions
    
    def enforce_upper_threshold_per_variable(self, actions, actions_indexable=None, states_indexable=None, var_name='total km'):
        var_index = self.indexes[var_name]
        
        if var_name in ['total km', 'hours alternative']:
            # Extract the variable from states 
            var_states = states_indexable[var_name]  # shape: (batch, days)
            max_value = tf.reduce_max(var_states[:,-28:], axis=1)  # shape: (batch,)
            # Add the upper threshold historic comparison value for the variable
            max_threshold = max_value + self.upper_thresholds_historic_comparison[var_name]  # shape: (batch,)
            # Create a tensor to broadcast the max_threshold to match the output shape
            max_threshold_tensor = tf.expand_dims(max_threshold, axis=1)  # shape: (batch, 1)
            max_threshold_tensor = tf.tile(max_threshold_tensor, [1, 7])  # shape: (batch, 7)
        elif var_name in ['km Z3-4', 'km Z5-T1-T2', 'km sprinting']:
            # Use the total km from actions to calculate the max threshold tensor
            total_km = actions_indexable['total km']  # shape: (batch, 7)
            max_threshold_tensor = total_km * self.upper_thresholds_ratios[var_name]  # shape: (batch, 7)
            
        elif var_name in ['strength training']:
            # Use the absolute threshold for strength training
            max_threshold = self.upper_thresholds_absolute[var_name]  # scalar
            # Broadcast the scalar to match the shape (batch, 7)
            max_threshold_tensor = tf.fill(tf.shape(actions)[:2], max_threshold)  # shape: (batch, 7)
        else:
            raise ValueError(f"Variable {var_name} not recognized")
        
        # Apply scaled tanh transformation to the specific variable in the actions tensor
        transformed = tf.where(
            actions[:, :, var_index] > 0,
            max_threshold_tensor * tf.tanh(actions[:, :, var_index] * 0.75),
            actions[:, :, var_index]
        )  # shape: (batch, 7)
        # Update the values in the actions tensor for the specified variable
        actions = tf.concat(
            [actions[:, :, :var_index], 
            tf.expand_dims(transformed, axis=2), 
            actions[:, :, var_index+1:]],
            axis=2
        )
        
        return actions
    
    def enforce_rest_days(self, actions, states_indexable):
        var_states = states_indexable['total km']
        total_km = actions[:, :, self.indexes['total km']]
        #Repalce zeros with inf to make sure they are not the lowest values
        min_km_value = tf.reduce_min(tf.where(var_states[:, -28:] != 0, var_states[:, -28:], float('inf')), axis=1)

        min_km_value = tf.expand_dims(min_km_value, axis=1)
        
        total_km = tf.where(total_km < min_km_value - self.lower_thresholds['total km'] * 0.5, tf.nn.relu(-total_km), total_km)

        actions = tf.concat(
            [
                actions[:, :, :self.indexes['total km']],
                tf.expand_dims(total_km, axis=2),
                actions[:, :, self.indexes['total km']+1:]
            ],
            axis=2
            )
        
        return actions

    
    def enforce_logical_correctness(self, actions):
        total_km = actions[:, :, self.indexes['total km']]
        
        # Apply the condition for km values
        km_z3_4 = actions[:, :, self.indexes['km Z3-4']]
        km_z5_t1_t2 = actions[:, :, self.indexes['km Z5-T1-T2']]
        km_sprinting = actions[:, :, self.indexes['km sprinting']]
        
        km_z3_4 = tf.where(total_km == 0, tf.nn.relu(-km_z3_4), km_z3_4)
        km_z5_t1_t2 = tf.where(total_km == 0, tf.nn.relu(-km_z5_t1_t2), km_z5_t1_t2)
        km_sprinting = tf.where(total_km == 0, tf.nn.relu(-km_sprinting), km_sprinting)
        
        # Update the actions tensor with the modified km values
        actions = tf.concat(
            [
                actions[:, :, :self.indexes['km Z3-4']],
                tf.expand_dims(km_z3_4, axis=2),
                tf.expand_dims(km_z5_t1_t2, axis=2),
                tf.expand_dims(km_sprinting, axis=2),
                actions[:, :, self.indexes['km sprinting']+1:]
            ],
            axis=2
        )
        
        return actions