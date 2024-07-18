import pytest
import numpy as np
from training_data import InputData, OutputData

def test_input_data():
    # Create hardcoded input data with values from 1 to 10 repeated across the batches and time steps
    data = np.tile(np.arange(1, 11), (5, 21, 1))

    # Initialize InputData object
    data_obj = InputData(data)

    # Test all variables for InputData
    variables = {
        'nr sessions': 1,
        'km total': 2,
        'km Z3-Z4': 3,
        'km Z5': 4,
        'km sprint': 5,
        'nr strength': 6,
        'hours alternative': 7,
        'exertion': 8,
        'recovery': 9,
        'training success': 10
    }

    for var, expected_value in variables.items():
        result = data_obj[var]
        expected = np.full((5, 21), expected_value)
        assert np.array_equal(result, expected), f"Failed on '{var}'"

def test_output_data():
    # Create hardcoded output data with values from 1 to 7 repeated across the batches and time steps
    data = np.tile(np.arange(1, 8), (5, 7, 1))

    # Initialize OutputData object
    data_obj = OutputData(data)

    # Test variables that should raise KeyError in OutputData
    output_only_vars = ['exertion', 'recovery', 'training success']
    for var in output_only_vars:
        with pytest.raises(KeyError):
            _ = data_obj[var]

    # Test valid output variables in OutputData
    variables = {
        'nr sessions': 1,
        'km total': 2,
        'km Z3-Z4': 3,
        'km Z5': 4,
        'km sprint': 5,
        'nr strength': 6,
        'hours alternative': 7
    }

    for var, expected_value in variables.items():
        result = data_obj[var]
        expected = np.full((5, 7), expected_value)
        assert np.array_equal(result, expected), f"Failed on '{var}'"