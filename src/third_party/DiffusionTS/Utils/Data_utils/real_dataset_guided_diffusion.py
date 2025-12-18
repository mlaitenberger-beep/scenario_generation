import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.io import arff
from scipy import stats
from copy import deepcopy
from torch.utils.data import Dataset
from Utils.masking_utils import noise_mask
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from sklearn.preprocessing import MinMaxScaler

class CustomDatasetGuided(Dataset):
    def __init__(
        self,
        data_root, 
        window=24, 
        save2npy=True, 
        neg_one_to_one=True,
        period='train',
        output_dir='./OUTPUT'
    ):
        super(CustomDatasetGuided, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'

        self.auto_norm, self.save2npy = neg_one_to_one, save2npy
        self.data_0, self.data_1, self.scaler = self.read_data(data_root, window)
        self.labels = np.zeros(self.data_0.shape[0] + self.data_1.shape[0]).astype(np.int64)
        self.labels[self.data_0.shape[0]:] = 1
        self.rawdata = np.vstack([self.data_0, self.data_1])
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)
        self.window, self.period = window, period
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]

        self.samples = self.normalize(self.rawdata)

        # np.save(os.path.join(self.dir, 'eeg_ground_0_truth.npy'), self.data_0)
        # np.save(os.path.join(self.dir, 'eeg_ground_1_truth.npy'), self.data_1)

        self.sample_num = self.samples.shape[0]

    def read_data(self, filepath, length):
        """
        Reads the data from the given filepath, removes outliers, classifies the data into two classes,
        and scales the data using MinMaxScaler.

        Args:
            filepath (str): Path to the .arff file containing the EEG data.
            length (int): Length of the window for classification.
        """
        df = pd.read_csv(filepath, header=0)
        data = df.values
        
        data = self.__create_windows__(data,length)
        df_0, df_1 = self.__Classify__(data, window=length)

        data_0 = df_0.reshape(-1, df_0.shape[-1])  # shape: (m * n, x)
        data_1 = df_1.reshape(-1, df_1.shape[-1])  # shape: (m * n, x) ######!!!!!!!! Möglicherweise fatal zusammengechoppt

        # print(f"Class 0: {data_0.shape}, Class 1: {data_1.shape}")

        data = np.vstack([data_0.reshape(-1, data_0.shape[-1]), data_1.reshape(-1, data_1.shape[-1])])

        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data_0, data_1, scaler
    @staticmethod
    def __create_windows__(series, window_size):
        """
        Slices a time series into non-overlapping sequences of a specified size.

        Args:
            series (np.ndarray): The input time series data (time_points, features).
            window_size (int): The desired length of each sequence.

        Returns:
            np.ndarray: An array of non-overlapping sequences (num_sequences, window_size, features).
                        Any leftover data points that don't form a full window are discarded.
        """
        num_time_points = series.shape[0]
        num_features = series.shape[1]
        num_sequences = num_time_points // window_size
  
        windows = series[:num_sequences * window_size].reshape(num_sequences, window_size, num_features)
        return windows

    @staticmethod
    def __Classify__(data, quantile_threshold=0.8, window=5):
        print('hi')
        """
        Klassifiziert Zeitreihensequenzen in "normal" und "extrem" basierend auf
        Volatilität, Drawdown und Z-Score der Renditen.

        Args:
            data (np.ndarray): 3D-Array (n_sequences, sequence_length, n_features)
            quantile_threshold (float): Quantil-Schwelle für Klassifikation.
            window (int): Fenstergröße für Rolling-Metriken.

        Returns:
            tuple: (normal_sequences, extreme_sequences)
        """
        if data.ndim != 3 or data.shape[0] == 0:
            raise ValueError("Daten müssen ein 3D-Array mit mindestens einer Sequenz sein.")

        n_seq, seq_len, n_feat = data.shape

        volatility_per_seq = []
        drawdown_per_seq = []
        zscore_per_seq = []

        for seq in data:
            df = pd.DataFrame(seq)

            # Returns berechnen
            #returns = df.pct_change().dropna()
            returns = df
            # Volatilität (Rolling, dann Mittelwert über Zeit & Features)
            rolling_vol = returns.rolling(window).std().mean().mean()

            # Drawdown
            rolling_max = df.cummax()
            drawdown = (df / rolling_max - 1).min().mean()

            # Z-Score
            mean_ret = returns.rolling(window).mean()
            std_ret = returns.rolling(window).std()
            zscores = ((returns - mean_ret) / std_ret).min().mean()

            volatility_per_seq.append(rolling_vol)
            drawdown_per_seq.append(drawdown)
            zscore_per_seq.append(zscores)

        # In Arrays umwandeln
        volatility_per_seq = np.array(volatility_per_seq)
        drawdown_per_seq = np.array(drawdown_per_seq)
        zscore_per_seq = np.array(zscore_per_seq)

        # Quantilschwellen berechnen
        thresholds = {
            'volatility': np.quantile(volatility_per_seq, quantile_threshold),
            'drawdown': np.quantile(drawdown_per_seq, 1 - quantile_threshold),
            'zscore': np.quantile(zscore_per_seq, 1 - quantile_threshold),
        }

        # Extremklassifikation: wenn eine Metrik gestresst ist
        is_extreme = (volatility_per_seq > thresholds['volatility']) | \
                    (drawdown_per_seq < thresholds['drawdown']) | \
                    (zscore_per_seq < thresholds['zscore'])

        # Aufteilen
        extreme_sequences = data[is_extreme]
        normal_sequences = data[~is_extreme]

        return normal_sequences, extreme_sequences

        #return np.array(normal_sequences, dtype=object), np.array(extreme_sequences, dtype=object)




    def __getitem__(self, ind):
        if self.period == 'test':
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            y = self.labels[ind]  # (1,) int
            return torch.from_numpy(x).float(), torch.tensor(y)

        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num

    def normalize(self, sq):
        d = self.__normalize(sq.reshape(-1, self.var_num))
        data = d.reshape(-1, self.window, self.var_num)
        return data

    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)
    
    def shift_period(self, period):
        assert period in ['train', 'test'], 'period must be train or test.'
        self.period = period