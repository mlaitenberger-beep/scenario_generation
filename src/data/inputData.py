
import numpy as np
import pandas as pd

class InputData:
    def __init__(self, data_hist, data_stress, dtype=np.float32):
        self.data_hist = pd.read_csv(data_hist).to_numpy(dtype=dtype)
        self.data_stress = pd.read_csv(data_stress).to_numpy(dtype=dtype)
       
    def relative_change(self, data):
        return (data[1:] - data[:-1]) / data[:-1]
    