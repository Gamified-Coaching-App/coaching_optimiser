import numpy as np
import tensorflow as tf

class TrainingData:
    def __init__(self, data):
        if not (isinstance(data, np.ndarray) or isinstance(data, tf.Tensor)):
            raise ValueError("Data must be a NumPy array.")
        self.data = data
        self.var_index_overarching = {
            'nr. sessions': 0,
            'total km': 1,
            'km Z3-4': 2,
            'km Z5-T1-T2': 3,
            'km sprinting': 4,
            'strength training': 5,
            'hours alternative': 6
        }
        self.var_index_output_only = {
            'perceived exertion': 7,
            'perceived trainingSuccess': 8,
            'perceived recovery': 9
        }
   
    def __getitem__(self, key):
        if key in self.var_index_overarching:
            return self.data[:, :, self.var_index_overarching[key]]
        elif key in self.var_index_output_only:
            return self.data[:, :, self.var_index_output_only[key]]
        else:
            raise KeyError(f"Invalid key. Valid keys are {list(self.var_index_overarching.keys())} and {list(self.var_index_output_only.keys())}")

class InputData(TrainingData):
    def __init__(self, data):
        if data.ndim != 3:
            raise ValueError("Input data must have 3 dimensions.")
        if data.shape[1] != 56:
            raise ValueError("The second dimension must be 56.")
        if data.shape[2] != 10:
            raise ValueError("The third dimension must be 10.")
        super().__init__(data)
    
    def __getitem__(self, key):
        return super().__getitem__(key)

class OutputData(TrainingData):
    def __init__(self, data):
        if data.ndim != 3:
            raise ValueError("Input data must have 3 dimensions.")
        if data.shape[1] != 7:
            raise ValueError("The second dimension must be 7.")
        if data.shape[2] != 7:
            raise ValueError("The third dimension must be 7.")
        super().__init__(data)

    def __getitem__(self, key):
        if key not in self.var_index_overarching:
            raise KeyError(f"Invalid key. Valid keys are: {list(self.var_index_overarching.keys())}")
        return super().__getitem__(key)