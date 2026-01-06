import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class InputData:
    def __init__(self, data_hist, data_stress, dtype=np.float32):
        self.data_hist = pd.read_csv(data_hist).to_numpy(dtype=dtype)
        self.data_stress = pd.read_csv(data_stress).to_numpy(dtype=dtype)
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

        # transform stress relative changes using the same scaler
        if self.data_stress_rel is not None:
            self.data_stress_rel_norm = self.normalize_to_neg_one_to_one(scaler.transform(self.data_stress_rel))
        else:
            self.data_stress_rel_norm = None
        return scaler
    

    def prepare_forcast_seq(self,stressed_features,stressed_seq_indices, len_historie):
        #Wie wird stress übergeben?: Erstmal Stress Werte auf selben Skala
        #Brauche letzten Bekannten Punkt. Dann in Relative Änderungen formulieren.
        forcast_seq = np.zeros_like(self.overlapping_sequences)
        forcast_seq[:len_historie,:,:] = self.overlapping_sequences[-len_historie:,:,:] 
        forcast_seq[:,stressed_features,stressed_seq_indices] = self.data_stress_rel_norm
        return forcast_seq

        
