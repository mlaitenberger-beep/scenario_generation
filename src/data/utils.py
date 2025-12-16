def create_overlapping_sequences(data, seq_length):
    num_samples = data.shape[0]
    num_features = data.shape[1]
    num_sequences = num_samples - seq_length + 1
    overlapping_sequences = np.zeros((num_sequences, seq_length, num_features))

    for i in range(num_sequences):
        overlapping_sequences[i, :, :] = data[i : i + seq_length, :]

    return overlapping_sequences

def data_processing(self, train_data):

    overlapping_data_relative_change = create_overlapping_sequences(
        train_data, 
        self.sequence_length  
    )
    data = overlapping_data_relative_change
    train = data[0:-1].reshape(-1, data.shape[-1])
    test = data[-1:,:,:]

    scaler = MinMaxScaler()
    train_scaled = normalize_to_neg_one_to_one(scaler.fit_transform(train))  # noqa: F821
    test_scaled = scaler.transform(test.reshape(-1, test.shape[-1])).reshape(test.shape)
    test_scaled = normalize_to_neg_one_to_one(test_scaled)  
    self.data = data
    self.scaler = scaler
    self.train_scaled = train_scaled
    self.test_scaled = test_scaled

    return data, train_scaled, test_scaled, scaler
def inject_stress(test_scaled, scaler, stressed_var, pred_length, stress_value,
                  normalize_to_neg_one_to_one=None, unnormalize_to_zero_to_one=None, noise_std=0.0):
    
            N, T, D = test_scaled.shape
            if isinstance(stressed_var, int):
                stressed_var = [stressed_var]

            # Handle stress_value: ensure it's a list corresponding to stressed_var
            if isinstance(stress_value, (int, float)):
                stress_values_for_vars = [stress_value] * len(stressed_var)
            elif isinstance(stress_value, (list, np.ndarray)):
                if len(stress_value) != len(stressed_var):
                    raise ValueError("If 'stress_value' is a list, its length must match 'stressed_var'.")
                stress_values_for_vars = stress_value
            else:
                raise TypeError("'stress_value' must be a float, an int, or a list/numpy array of floats/ints.")

            # Handle noise_std: ensure it's a list corresponding to stressed_var
            if isinstance(noise_std, (int, float)):
                noise_stds_for_vars = [noise_std] * len(stressed_var)
            elif isinstance(noise_std, (list, np.ndarray)):
                if len(noise_std) != len(stressed_var):
                    raise ValueError("If 'noise_std' is a list, its length must match 'stressed_var'.")
                noise_stds_for_vars = noise_std
            else:
                raise TypeError("'noise_std' must be a float, an int, or a list/numpy array of floats/ints.")

            if normalize_to_neg_one_to_one is None:
                normalize_to_neg_one_to_one = lambda x: x
            if unnormalize_to_zero_to_one is None:
                unnormalize_to_zero_to_one = lambda x: x

            data_01 = unnormalize_to_zero_to_one(test_scaled.reshape(-1, D)).reshape(N, T, D)
            real_values = scaler.inverse_transform(data_01.reshape(-1, D)).reshape(N, T, D)

            for i, var in enumerate(stressed_var):
                noise = np.random.normal(0, noise_stds_for_vars[i], real_values[:, :pred_length, var].shape)
                real_values[:, :pred_length, var] = stress_values_for_vars[i] + noise

            data_01_stressed = scaler.transform(real_values.reshape(-1, D)).reshape(N, T, D)
            stressed_scaled = normalize_to_neg_one_to_one(data_01_stressed)

            return stressed_scaled

def relative_change_to_original(relative_changes, initial_values):
    original_values = np.zeros_like(relative_changes)

    original_values[:, 0, :] = initial_values * (relative_changes[:, 0, :] +1)

    for i in range(1, relative_changes.shape[1]):
        original_values[:, i, :] = original_values[:, i-1, :] * (relative_changes[:, i, :]+1 )
    return original_values