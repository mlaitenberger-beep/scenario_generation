import os
import torch
import numpy as np
import sys
from src.third_party.DiffusionTS.engine.solver import Trainer
from src.third_party.DiffusionTS.Data.build_dataloader import build_dataloader
from src.third_party.DiffusionTS.Utils.io_utils import load_yaml_config, instantiate_from_config
from sklearn.preprocessing import MinMaxScaler
from src.adapters.modelAdapter import ModelAdapter
from src.third_party.DiffusionTS.torch.utils.data import Dataset, DataLoader
from src.data.utils import CustomDataset, Args_Example

class Diffusion_ts_adapter(ModelAdapter):
    def __init__(self, diffusionTS_config_path, config, data_hist, data_forcast):
        self.diffusionTS_config_path =  diffusionTS_config_path 
        #Daten kommen pre-processed rein
        self.data_hist = data_hist
        self.data_forcast = data_forcast

    def create_model_config(self, config):
       
        diffusionTS_config = load_yaml_config(self.diffusionTS_config_path)
        diffusionTS_config['model']['params']['sequence_length'] = config['sequence_length']
        diffusionTS_config['data']['params']['feature_size'] = config['feature_size']  
        diffusionTS_config['solver']['results_folder'] = config['results_folder']
        diffusionTS_config['dataloader']['train_dataset']['params']['data_root'] = config['data_root']
        diffusionTS_config['dataloader']['test_dataset']['params']['data_root'] = config['data_root']
        #Hier sollte die Diffusion TS spezifische config komplett befüllt sein.
        return diffusionTS_config
    

    def load_model(self):

        hist_dataset = CustomDataset(data = self.data_hist)
        dataloader = DataLoader(hist_dataset, batch_size=self.config['batch_size'], shuffle=False,num_workers=0, drop_last=True, pin_memory=True, sampler=None)
        diffusionTS_config = load_yaml_config(create_model_config(self, self.config))
        args =  Args_Example(self.diffusionTS_config_path, self.config['results_folder'])
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        model = instantiate_from_config(diffusionTS_config['model']).to(device)
        trainer = Trainer(config=diffusionTS_config, args=args, model=model, dataloader={'dataloader':dataloader})
        return trainer 
        
    def train(self, trainer):
        if self.config['milestone'] != None:
            trainer.load(milestone = self.config['milestone'])
        else:
            trainer.load()
        return trainer 

    def predict(self, trainer):
        seq_length, feat_num = self.config['seq_length'] , self.config['feature_size']
        #Hier Stress-Sequenz erstellen 
        forcast_dataset = CustomDataset(self.data_forcast, regular=False, stressed_var= self.config['stressed_features'])
        forcast_dataloader = torch.utils.data.DataLoader(forcast_dataset, batch_size=self.data_forcast.shape[0], shuffle=True, num_workers=0, pin_memory=True, sampler=None)
        samples = []
        for _ in range(self.config['num_samples']):
            sample, *_ = trainer.restore(forcast_dataloader, shape=[seq_length, feat_num], sampling_steps= self.config['sampling_steps'])
            samples.append(sample)
        sample = np.concatenate(samples, axis=0)

        return sample