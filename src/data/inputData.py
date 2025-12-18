
class InputData:
    def __init__(self, data_hist, data_stress, dtype=np.float32):
        self.data_hist = pd.read_csv(data_hist).to_numpy(dtype=dtype)
        self.data_stress = pd.read_csv(data_stress).to_numpy(dtype=dtype)
       
    def relative_change(self, data):
        return (data[1:] - data[:-1]) / data[:-1]
    
    def split_historie_forcast(self):
        return self.data_hist[:-1,:] ,  self.data_hist[-1:,:] 


    def create_overlapping_sequences(data, seq_length):
        num_samples = data.shape[0]
        num_features = data.shape[1]
        num_sequences = num_samples - seq_length + 1
        overlapping_sequences = np.zeros((num_sequences, seq_length, num_features))

        for i in range(num_sequences):
            overlapping_sequences[i, :, :] = data[i : i + seq_length, :]

        return overlapping_sequences
    
    def normalize(self, data):
        scaler = MinMaxScaler()
        scaler.fit(data)
        data_scaled = scaler.transform(data)
        return data_scaled, scaler 
    
    def normalize_to_neg_one_to_one(x):
        return x * 2 - 1
