import tempfile
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.inputData import InputData

def write_csv(arr, path):
    pd.DataFrame(arr).to_csv(path, index=False)

def main():
    with tempfile.TemporaryDirectory() as d:
        hist = np.vstack([np.arange(3) + i for i in range(10)])
        stress = np.vstack([np.arange(3) + 100 + i for i in range(4)])
        hist_path = os.path.join(d, 'hist.csv')
        stress_path = os.path.join(d, 'stress.csv')
        write_csv(hist, hist_path)
        write_csv(stress, stress_path)

        idata = InputData(hist_path, stress_path)
        idata.relative_change()
        scaler = idata.normalize()
        idata.create_overlapping_sequences(seq_length=4)
        fc = idata.prepare_forcast_seq(stressed_features=[0], stressed_seq_indices=[0], len_historie=2)
        print('InputData smoke test passed; overlapping shape:', idata.overlapping_sequences.shape)

if __name__ == '__main__':
    main()
