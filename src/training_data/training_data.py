import numpy as np

class TrainingData:
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a NumPy array.")
        self.data = data
        self.var_index_overarching = {
            'numberSessions': 0,
            'kmTotal': 1,
            'kmZ3Z4': 2,
            'kmZ5': 3,
            'kmSprint': 4,
            'numberStrengthSessions': 5,
            'hoursAlternative': 6
        }
        self.var_index_output_only = {
            'perceivedExertion': 7,
            'perceivedRecovery': 8,
            'perceivedTrainingSuccess': 9
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
        if data.shape[1] != 21:
            raise ValueError("The second dimension must be 21.")
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