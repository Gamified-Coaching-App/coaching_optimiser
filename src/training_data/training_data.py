import numpy as np
import tensorflow as tf

"""
TrainingData class to have consistent data access interface for input and output data of the model (states and actions)
"""
class TrainingData:
    def __init__(self, data):
        if not (isinstance(data, np.ndarray) or isinstance(data, tf.Tensor)):
            raise ValueError("Data must be a NumPy array.")
        self.data = data
   
    def __getitem__(self, index):
        return self.data[:, :, index]

"""
InputData class as child class for states (10 variables)
"""   
class InputData(TrainingData):
    def __init__(self, data):
        if data.ndim != 3:
            raise ValueError("Input data must have 3 dimensions.")
        if data.shape[1] != 56:
            raise ValueError("The second dimension must be 56.")
        if data.shape[2] != 10:
            raise ValueError("The third dimension must be 10.")
        self.var_index_input = {
            'nr. sessions': 0,
            'total km': 1,
            'km Z3-4': 2,
            'km Z5-T1-T2': 3,
            'km sprinting': 4,
            'strength training': 5,
            'hours alternative': 6,
            'perceived exertion': 7,
            'perceived trainingSuccess': 8,
            'perceived recovery': 9
        }
        super().__init__(data)
    
    def __getitem__(self, key):
        if key not in self.var_index_input:
            raise KeyError(f"Invalid key. Valid keys are: {list(self.var_index_input())}")
        return super().__getitem__(self.var_index_input[key])

"""
OutputData class as child class for actions (4 variables)
"""
class OutputData(TrainingData):
    def __init__(self, data):
        if data.ndim != 3:
            raise ValueError("Input data must have 3 dimensions.")
        if data.shape[1] != 7:
            raise ValueError("The second dimension must be 7.")
        if data.shape[2] != 4:
            raise ValueError("The third dimension must be 4.")
        self.var_index_output = {
            'total km': 0,
            'km Z3-4': 1,
            'km Z5-T1-T2': 2,
            'km sprinting': 3
        }
        super().__init__(data)

    def __getitem__(self, key):
        if key not in self.var_index_output:
            raise KeyError(f"Invalid key. Valid keys are: {list(self.var_index_output())}")
        return super().__getitem__(self.var_index_output[key])