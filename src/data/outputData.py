class OutputData:
    def __init__(self,results,config):
        self.results = results
        self.configs = config

    def unnormalize_to_zero_to_one(x):
        return (x + 1) * 0.5
    
    def re_normalize(self,scaler, data):
        return scaler.inverse_transform(data)

    def relative_change_to_original(self, relative_changes, initial_values):
        original_values = np.zeros_like(relative_changes)
        original_values[:, 0, :] = initial_values * (relative_changes[:, 0, :] +1)
        for i in range(1, relative_changes.shape[1]):
            original_values[:, i, :] = original_values[:, i-1, :] * (relative_changes[:, i, :]+1 )
        return original_values
    

