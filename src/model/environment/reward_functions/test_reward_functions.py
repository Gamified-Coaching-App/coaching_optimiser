import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
src_path = os.path.abspath(os.path.join(current_dir, '../../../'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import pytest
import numpy as np
import tensorflow as tf
from training_data.training_data import InputData, OutputData
import h5py
from model.environment.reward_functions.utils.utils import get_absolute_values, smooth_count
import json
import pandas as pd

"""Prepare test files: Original data, processed data, min_max_values"""
original_data_full = pd.read_csv('../../../../data/day_approach_maskedID_timeseries.csv')
preprocessed_data_first_two_columns = pd.read_csv('../../data/first_two_rows.csv')
preprocessed_data_first_two_columns = preprocessed_data_first_two_columns.drop(columns=['Date', 'Athlete ID','injury'], errors='ignore')
full_data = pd.read_csv('../../../../data/day_approach_maskedID_timeseries.csv')

with h5py.File('../../data/processed_data.h5', 'r') as hf:
    test_states = tf.convert_to_tensor(hf['X_train'][:][:2], dtype=tf.float32)
with open('../../data/min_max_values.json', 'r') as file:
    min_max_values = json.load(file)
test_states = InputData(test_states)

def test_get_absolute_values_nr_sessions():
    """Test transformation of 'nr. sessions' from normalized to absolute values."""
    expected_output = tf.convert_to_tensor(preprocessed_data_first_two_columns.iloc[:, 0::10].values, dtype=tf.float32)
    absolute_values = get_absolute_values(test_states['nr. sessions'], min_max_values, 'nr. sessions')
    original_data_0 = tf.convert_to_tensor(original_data_full['nr. sessions'][:56].values, dtype=tf.float32)
    original_data_1 = tf.convert_to_tensor(original_data_full['nr. sessions'][1:57].values, dtype=tf.float32)
    tf.debugging.assert_near(absolute_values, expected_output, atol=0.001)
    tf.debugging.assert_equal(expected_output[0], original_data_0)
    tf.debugging.assert_equal(expected_output[1], original_data_1)

def test_get_absolute_values_total_km():
    """Test transformation of 'total km' from normalized to absolute values."""
    absolute_values = get_absolute_values(test_states['total km'], min_max_values, 'total km')
    expected_output = tf.convert_to_tensor(preprocessed_data_first_two_columns.iloc[:, 1::10].values, dtype=tf.float32)
    original_data_0 = tf.convert_to_tensor(original_data_full['total km'][:56].values, dtype=tf.float32)
    original_data_1 = tf.convert_to_tensor(original_data_full['total km'][1:57].values, dtype=tf.float32)
    tf.debugging.assert_near(absolute_values, expected_output, atol=0.001)
    tf.debugging.assert_equal(expected_output[0], original_data_0)
    tf.debugging.assert_equal(expected_output[1], original_data_1)

def test_get_absolute_values_km_Z34():
    """Test transformation of 'km Z3-4' from normalized to absolute values."""
    absolute_values = get_absolute_values(test_states['km Z3-4'], min_max_values, 'km Z3-4')
    expected_output = tf.convert_to_tensor(preprocessed_data_first_two_columns.iloc[:, 2::10].values, dtype=tf.float32)
    original_data_0 = tf.convert_to_tensor(original_data_full['km Z3-4'][:56].values, dtype=tf.float32)
    original_data_1 = tf.convert_to_tensor(original_data_full['km Z3-4'][1:57].values, dtype=tf.float32)
    tf.debugging.assert_near(absolute_values, expected_output, atol=0.001)
    tf.debugging.assert_equal(expected_output[0], original_data_0)
    tf.debugging.assert_equal(expected_output[1], original_data_1)

def test_get_absolute_values_km_Z5_T1_T2():
    """Test transformation of 'km Z5-T1-T2' from normalized to absolute values."""
    absolute_values = get_absolute_values(test_states['km Z5-T1-T2'], min_max_values, 'km Z5-T1-T2')
    expected_output = tf.convert_to_tensor(preprocessed_data_first_two_columns.iloc[:, 3::10].values, dtype=tf.float32)
    original_data_0 = tf.convert_to_tensor(original_data_full['km Z5-T1-T2'][:56].values, dtype=tf.float32)
    original_data_1 = tf.convert_to_tensor(original_data_full['km Z5-T1-T2'][1:57].values, dtype=tf.float32)
    tf.debugging.assert_near(absolute_values, expected_output, atol=0.001)
    tf.debugging.assert_equal(expected_output[0], original_data_0)
    tf.debugging.assert_equal(expected_output[1], original_data_1)

def test_get_absolute_values_km_sprinting():
    """Test transformation of 'km sprinting' from normalized to absolute values."""
    absolute_values = get_absolute_values(test_states['km sprinting'], min_max_values, 'km sprinting')
    expected_output = tf.convert_to_tensor(preprocessed_data_first_two_columns.iloc[:, 4::10].values, dtype=tf.float32)
    original_data_0 = tf.convert_to_tensor(original_data_full['km sprinting'][:56].values, dtype=tf.float32)
    original_data_1 = tf.convert_to_tensor(original_data_full['km sprinting'][1:57].values, dtype=tf.float32)
    tf.debugging.assert_near(absolute_values, expected_output, atol=0.001)
    tf.debugging.assert_equal(expected_output[0], original_data_0)
    tf.debugging.assert_equal(expected_output[1], original_data_1)

def test_get_absolute_values_strength_training():
    """Test transformation of 'strength training' from normalized to absolute values."""
    absolute_values = get_absolute_values(test_states['strength training'], min_max_values, 'strength training')
    expected_output = tf.convert_to_tensor(preprocessed_data_first_two_columns.iloc[:, 5::10].values, dtype=tf.float32)
    original_data_0 = tf.convert_to_tensor(original_data_full['strength training'][:56].values, dtype=tf.float32)
    original_data_1 = tf.convert_to_tensor(original_data_full['strength training'][1:57].values, dtype=tf.float32)
    tf.debugging.assert_near(absolute_values, expected_output, atol=0.001)
    tf.debugging.assert_equal(expected_output[0], original_data_0)
    tf.debugging.assert_equal(expected_output[1], original_data_1)

def test_get_absolute_values_hours_alternative():
    """Test transformation of 'hours alternative' from normalized to absolute values."""
    absolute_values = get_absolute_values(test_states['hours alternative'], min_max_values, 'hours alternative')
    expected_output = tf.convert_to_tensor(preprocessed_data_first_two_columns.iloc[:, 6::10].values, dtype=tf.float32)
    original_data_0 = tf.convert_to_tensor(original_data_full['hours alternative'][:56].values, dtype=tf.float32)
    original_data_1 = tf.convert_to_tensor(original_data_full['hours alternative'][1:57].values, dtype=tf.float32)
    tf.debugging.assert_near(absolute_values, expected_output, atol=0.001)
    tf.debugging.assert_equal(expected_output[0], original_data_0)
    tf.debugging.assert_equal(expected_output[1], original_data_1)

def test_get_absolute_values_perceived_exertion():
    """Test transformation of 'perceived exertion' from normalized to absolute values."""
    absolute_values = get_absolute_values(test_states['perceived exertion'], min_max_values, 'perceived exertion')
    expected_output = tf.convert_to_tensor(preprocessed_data_first_two_columns.iloc[:, 7::10].values, dtype=tf.float32)
    original_data_0 = tf.convert_to_tensor(original_data_full['perceived exertion'][:56].values, dtype=tf.float32)
    original_data_1 = tf.convert_to_tensor(original_data_full['perceived exertion'][1:57].values, dtype=tf.float32)
    tf.debugging.assert_near(absolute_values, expected_output, atol=0.001)
    tf.debugging.assert_equal(expected_output[0], original_data_0)
    tf.debugging.assert_equal(expected_output[1], original_data_1)

def test_get_absolute_values_perceived_trainingSuccess():
    """Test transformation of 'perceived trainingSuccess' from normalized to absolute values."""
    absolute_values = get_absolute_values(test_states['perceived trainingSuccess'], min_max_values, 'perceived trainingSuccess')
    expected_output = tf.convert_to_tensor(preprocessed_data_first_two_columns.iloc[:, 8::10].values, dtype=tf.float32)
    original_data_0 = tf.convert_to_tensor(original_data_full['perceived trainingSuccess'][:56].values, dtype=tf.float32)
    original_data_1 = tf.convert_to_tensor(original_data_full['perceived trainingSuccess'][1:57].values, dtype=tf.float32)
    tf.debugging.assert_near(absolute_values, expected_output, atol=0.001)
    tf.debugging.assert_equal(expected_output[0], original_data_0)
    tf.debugging.assert_equal(expected_output[1], original_data_1)

def test_get_absolute_values_perceived_recovery():
    """Test transformation of 'perceived recovery' from normalized to absolute values."""
    absolute_values = get_absolute_values(test_states['perceived recovery'], min_max_values, 'perceived recovery')
    expected_output = tf.convert_to_tensor(preprocessed_data_first_two_columns.iloc[:, 9::10].values, dtype=tf.float32)
    original_data_0 = tf.convert_to_tensor(original_data_full['perceived recovery'][:56].values, dtype=tf.float32)
    original_data_1 = tf.convert_to_tensor(original_data_full['perceived recovery'][1:57].values, dtype=tf.float32)
    tf.debugging.assert_near(absolute_values, expected_output, atol=0.001)
    tf.debugging.assert_equal(expected_output[0], original_data_0)
    tf.debugging.assert_equal(expected_output[1], original_data_1)

def print_tensor_for_code(tensor):
    tensor_np = tensor.numpy()  # Convert tensor to numpy array
    tensor_list = tensor_np.tolist()  # Convert numpy array to list
    print("[")
    for sublist in tensor_list:
        print("[", end="")
        print(", ".join(f"{value:.2f}" for value in sublist), end="")
        print("],")
    print("]")


def test_smooth_count():
    input = tf.constant([[0.0, 0.1, 1.0], [0.0, 0.1, 0.05]], dtype=tf.float32)
    result= smooth_count(input)
    print("count result", result)
    tf.debugging.assert_equal(False, True)


# # Check if gradient flow is maintained
# with tf.GradientTape() as tape:
#     tape.watch(input_data)
#     result = tf.reduce_sum(inputs['nr. sessions']) + tf.reduce_sum(outputs['total km'])

# # Compute gradients
# grads = tape.gradient(result, input_data)
# print("Gradients computed successfully:", grads is not None)

if __name__ == "__main__":
    pytest.main([__file__, '-s'])