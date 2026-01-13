import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch



class CustomDataset(Dataset):
    def __init__(self, data, regular=True, stressed_var=None):
        super(CustomDataset, self).__init__()
        self.sample_num = data.shape[0]
        self.samples = data
        self.regular = regular
        self.mask = np.ones_like(data)
        if not self.regular:
            self.mask[:, :, :] = 0.
            #Hier bekannte Punkte Variable einfügen
            self.mask[:, :, stressed_var] = 1.
        self.mask = self.mask.astype(bool)

    def __getitem__(self, ind):
        x = self.samples[ind, :, :]
        if self.regular:
            return torch.from_numpy(x).float()
        mask = self.mask[ind, :, :]
        return torch.from_numpy(x).float(), torch.from_numpy(mask)

    def __len__(self):
        return self.sample_num
    

class Args_Example:
    def __init__(self, config_path, results_folder):
        self.config_path = config_path
        self.results_folder = results_folder
        self.gpu = 0
        os.makedirs(self.results_folder, exist_ok=True)