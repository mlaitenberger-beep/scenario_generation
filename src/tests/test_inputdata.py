import os
import tempfile
import numpy as np
import pandas as pd
import sys
import os

# ensure src is on path for imports when running tests from workspace root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.inputData import InputData


def _write_csv(arr, path):
    pd.DataFrame(arr).to_csv(path, index=False)


def test_inputdata_flow(tmp_path):
    # create simple historic and stress csv files
    hist = np.vstack([np.arange(3) + i for i in range(10)])  # shape (10,3)
    stress = np.vstack([np.arange(3) + 100 + i for i in range(4)])  # shape (4,3)

    hist_path = tmp_path / "hist.csv"
    stress_path = tmp_path / "stress.csv"
    _write_csv(hist, hist_path)
    _write_csv(stress, stress_path)

    idata = InputData(str(hist_path), str(stress_path))
    idata.relative_change()

    assert idata.data_hist_rel.shape == (9, 3)
    assert idata.data_stress_rel.shape[0] == 4

    scaler = idata.normalize()
    assert scaler is not None
    assert idata.data_hist_rel_norm.min() >= -1.0 - 1e-6
    assert idata.data_hist_rel_norm.max() <= 1.0 + 1e-6

    seqs = idata.create_overlapping_sequences(seq_length=4)
    assert seqs.shape[1] == 4

    fc = idata.prepare_forcast_seq(stressed_features=[0], stressed_seq_indices=[0], len_historie=2)
    assert fc.shape == seqs.shape
