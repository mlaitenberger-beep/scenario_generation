import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class InputData:
    def __init__(self, data_hist, data_stress):
        # Read CSVs with first row as header; convert to numpy arrays with float dtype
        self.data_hist = pd.read_csv(data_hist, header=0, dtype=float).to_numpy()
        self.data_stress = pd.read_csv(data_stress, header=0, dtype=float).to_numpy()
        self.data_hist_rel = None
        self.data_stress_rel = None #Stress doch direkt als rel änderung?!
        self.data_hist_rel_norm = None
        self.data_stress_rel_norm = None
        self.overlapping_sequences = None


    def relative_change(self, eps=1e-12):
        hist_prev = self.data_hist[:-1]
        self.data_hist_rel = (self.data_hist[1:] - hist_prev) / (hist_prev + eps)

        # Stress: erster Schritt knüpft an letzten historischen Wert an
        last_hist = self.data_hist[-1]             # Shape: (features,) oder scalar
        stress_prev = self.data_stress[:-1]
        stress_first_prev = last_hist              # Referenz für t=0 im Stress

        # rel change für Stress:
        # - erstes Stress-Element relativ zu last_hist
        first_rel = (self.data_stress[0] - stress_first_prev) / (stress_first_prev + eps)

        # - restliche Stress-Elemente relativ zu vorherigem Stresswert
        rest_rel = (self.data_stress[1:] - stress_prev) / (stress_prev + eps)

        # zusammensetzen -> Länge = len(stress)
        self.data_stress_rel = np.vstack([first_rel, rest_rel])



    def create_overlapping_sequences(self, seq_length):
        num_samples = self.data_hist_rel_norm.shape[0]
        num_features = self.data_hist_rel_norm.shape[1]
        num_sequences = num_samples - seq_length + 1
      

        overlapping_sequences = np.zeros((num_sequences, seq_length, num_features))

        for i in range(num_sequences):
            overlapping_sequences[i, :, :] = self.data_hist_rel_norm[i : i + seq_length, :]

        self.overlapping_sequences = overlapping_sequences
        return self.overlapping_sequences
    
    @staticmethod
    def normalize_to_neg_one_to_one(x):
        return x * 2 - 1
    
    def normalize(self):
        #Alles einfacher wenn stress direkt als änderungen übergeben werden
        scaler = MinMaxScaler()
        scaler.fit(self.data_hist_rel)
        data_scaled = scaler.transform(self.data_hist_rel)
        data_scaled = self.normalize_to_neg_one_to_one(data_scaled)
        self.data_hist_rel_norm = data_scaled

        self.data_stress_rel_norm = self.normalize_to_neg_one_to_one(scaler.transform(self.data_stress_rel))
        
        return scaler
    

    def prepare_forcast_seq(self, stressed_features, stressed_seq_indices, len_historie, seq_length=None):
        """Create a forecast input sequence with stress values inserted.

        - stressed_features: list of feature indices to stress
        - stressed_seq_indices: either a single start index (int or single-item list) or a list of indices where stress values map
        - len_historie: number of last historical steps to copy into the beginning of the forecast
        - seq_length: total forecast sequence length (optional, defaults to overlap width)
        Returns array shape (1, seq_length, n_features)
        """
        n_features = self.data_hist_rel_norm.shape[1]
        
        # If len_historie is None and we have overlapping sequences, default to half the sequence length
        if len_historie is None and self.overlapping_sequences is not None:
            len_historie = self.overlapping_sequences.shape[1] // 2
        
        if seq_length is None:
            # try to derive from existing overlapping sequences, else use len_historie + stress length
            if self.overlapping_sequences is not None:
                seq_length = self.overlapping_sequences.shape[1]
            else:
                if len_historie is None:
                    raise ValueError("len_historie must be provided when overlapping_sequences is None")
                seq_length = len_historie + (self.data_stress_rel.shape[0] if self.data_stress_rel is not None else 0)

        forcast_seq = np.zeros((1, seq_length, n_features))
        # copy last len_historie steps from the most recent overlapping sequence
        if self.overlapping_sequences is not None and len_historie is not None:
            last_hist = self.overlapping_sequences[-1, -len_historie:, :]
            forcast_seq[0, :len_historie, :] = last_hist

        # nothing to insert if no stress provided
        if self.data_stress_rel is None:
            return forcast_seq
        
        # validate required params for stressing
        if stressed_seq_indices is None:
            raise ValueError("stressed_seq_indices is required when data_stress is provided")
        if stressed_features is None:
            raise ValueError("stressed_features is required when data_stress is provided")

        stress = self.data_stress_rel  # shape (T_s, n_features)
        T_s = stress.shape[0]

        # treat single start index
        if isinstance(stressed_seq_indices, (list, tuple)) and len(stressed_seq_indices) == 1:
            start = stressed_seq_indices[0]
            for t in range(min(T_s, seq_length - start)):
                for f in stressed_features:
                    forcast_seq[0, start + t, f] = stress[t, f]
        else:
            # map given indices elementwise
            for k, idx in enumerate(stressed_seq_indices):
                if k >= T_s or idx >= seq_length:
                    continue
                for f in stressed_features:
                    forcast_seq[0, idx, f] = stress[k, f]
        return forcast_seq

        
