from src.adapters.modelAdapter import ModelAdapter

# src/adapters/diffusion_ts_adapter.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler



class CustomDataset(Dataset):
    def __init__(self, data, regular=True, stressed_var=None):
        super(CustomDataset, self).__init__()
        self.sample_num = data.shape[0]
        self.samples = data
        self.regular = regular
        self.mask = np.ones_like(data)
        if not self.regular:
            self.mask[:, :, :] = 0.
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


class Diffusion_ts_adapter(ModelAdapter):
    def __init__(self, config, model, data_train, sequnece_length):
        self.config = config
        self.model = model 
        self.train_data = data_train    
        self.sequence_length = sequnece_length 

    

    def load_model(self):

        data = getattr(self, "data", None)
        train_scaled = getattr(self, "train_scaled", None)

        train_dataset = CustomDataset(train_scaled.reshape(#länge data, -1, data.shape[-1]))
        dataloader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            drop_last=True,
            pin_memory=True,
            sampler=None
        )

        class Args_Example:
            def __init__(self) -> None:
                self.config_path = #config_path
                self.results_folder = #results/folder/path
                self.gpu = 0
                os.makedirs(self.results_folder, exist_ok=True)

        args = Args_Example()

        
              
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        model = instantiate_from_config(configs['model']).to(device)  
        trainer = Trainer(config=configs, args=args, model=model, dataloader={'dataloader': dataloader}) 
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.configs = configs
        self.args = args
        self.trainer = trainer
        
    def train(self,milestone=None):
        if milestone != None:
            self.trainer.load(milestone = milestone)
            self.trainer.train()
        else
            self.trainer.train()
        

    def predict(self):
        _, seq_length, feat_num =  last_sequence.shape

        test_dataset = MacroDataset(last_sequence, regular=False, stressed_var= stressed_var)
        real = scaler.inverse_transform(unnormalize_to_zero_to_one(last_sequence.reshape(-1, feat_num))).reshape(last_sequence.shape)
        stress_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=last_sequence.shape[0], shuffle=True, num_workers=0, pin_memory=True, sampler=None)

        samples = []
        for _ in range(size):
            sample, *_ = trainer.restore(stress_dataloader, shape=[seq_length, feat_num], coef=1e-1, stepsize=1e-1, sampling_steps=200)
            samples.append(sample)
        sample = np.concatenate(samples, axis=0)
        sample = scaler.inverse_transform(unnormalize_to_zero_to_one(sample.reshape(-1, feat_num))).reshape(-1, unseen.shape[1], unseen.shape[2])
        return sample

    def create_model_config(self, config):
        diffusion_config_path = #
        diffusion_config = load_yaml_config(args.diffusion_config_path)
        diffusion_config['model']['params']['sequence_length'] = config['sequence_length']
        diffusion_config['data']['params']['feature_size'] = config['feature_size']  
        diffusion_config['solver']['results_folder'] = config['results_folder']
        diffusion_config['dataloader']['train_dataset']['params']['data_root'] = config['data_root']
        diffusion_config['dataloader']['test_dataset']['params']['data_root'] = config['data_root']

        return diffusion_config