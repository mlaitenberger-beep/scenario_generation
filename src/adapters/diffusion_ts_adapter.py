import os
import torch
import numpy as np
import sys
from data.inputData import InputData


from src.third_party.DiffusionTS.engine.solver import Trainer
from src.third_party.DiffusionTS.Data.build_dataloader import build_dataloader
from src.third_party.DiffusionTS.Utils.io_utils import load_yaml_config, instantiate_from_config
from src.adapters.modelAdapter import ModelAdapter
from src.third_party.DiffusionTS.torch.utils.data import Dataset, DataLoader
from src.data.utils import CustomDataset, Args_Example

class Diffusion_ts_adapter(ModelAdapter):
    def __init__(self, diffusionTS_config_path=None, config=None):
        # allow default path if none provided
        if diffusionTS_config_path is None:
            diffusionTS_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'diffusion_configs', 'diffusion_config.yaml'))
        self.diffusionTS_config_path = diffusionTS_config_path
        self.config = config or {}
        # Data placeholders
        self.data_hist = None
        self.forcast_seq = None
        self.scalar = None


    def create_model_config(self):
        diffusionTS_config = load_yaml_config(self.diffusionTS_config_path)
        # set model and data related params from the higher-level config
        seq = self.config.get('seq_length') or self.config.get('sequence_length') or self.config.get('seq_length')
        if seq is not None:
            diffusionTS_config['model']['params']['seq_length'] = seq
            diffusionTS_config['model']['params']['sequence_length'] = seq

        feat = self.config.get('feature_size')
        if feat is not None:
            diffusionTS_config['model']['params']['feature_size'] = feat
            diffusionTS_config['data']['params']['feature_size'] = feat

        if self.config.get('results_folder'):
            diffusionTS_config['solver']['results_folder'] = self.config.get('results_folder')

        if self.config.get('data_root'):
            diffusionTS_config['dataloader']['train_dataset']['params']['data_root'] = self.config.get('data_root')
            diffusionTS_config['dataloader']['test_dataset']['params']['data_root'] = self.config.get('data_root')

        return diffusionTS_config
    

    def load_model(self):
        # build dataloader from prepared historical sequences
        hist_dataset = CustomDataset(data=self.data_hist)
        dataloader = DataLoader(hist_dataset, batch_size=self.config.get('batch_size', 64), shuffle=False, num_workers=0, drop_last=True, pin_memory=True, sampler=None)

        diffusionTS_config = self.create_model_config()
        args = Args_Example(self.diffusionTS_config_path, self.config.get('results_folder', './results'))

        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        model = instantiate_from_config(diffusionTS_config['model']).to(device)
        trainer = Trainer(config=diffusionTS_config, args=args, model=model, dataloader={'dataloader': dataloader})
        return trainer
        
    def train(self, trainer):
        # If milestone provided, resume; otherwise run training
        if self.config.get('milestone') is not None:
            trainer.load(milestone=self.config['milestone'])
        else:
            trainer.train()
        return trainer

    def predict(self, trainer):
        seq_length = self.config.get('seq_length')
        feat_num = self.config.get('feature_size')
        if self.forcast_seq is None:
            raise ValueError("forecast sequence not prepared. call data_input() first")

        num_samples = self.config.get('num_samples', 100)
        pred = np.repeat(self.forcast_seq, num_samples, axis=0)
        forcast_dataset = CustomDataset(pred, regular=False, stressed_var=self.config.get('stressed_features'))
        # batch all samples together
        forcast_dataloader = torch.utils.data.DataLoader(forcast_dataset, batch_size=pred.shape[0], shuffle=False, num_workers=0, pin_memory=True, sampler=None)

        samples, reals, masks = trainer.restore(forcast_dataloader, shape=[seq_length, feat_num], sampling_steps=self.config.get('sampling_steps', 200))
        return samples
    
    def data_input(self):
        inputData = InputData(self.config.get('data_hist_path'), self.config.get('data_stress_path'))
        inputData.relative_change()
        self.scalar = inputData.normalize()
        # prepare overlapping sequences of length seq_length
        inputData.create_overlapping_sequences(self.config.get('seq_length'))
        self.data_hist = inputData.overlapping_sequences
        self.forcast_seq = inputData.prepare_forcast_seq(self.config.get('stressed_features'), self.config.get('stressed_seq_indices'), self.config.get('len_historie'))

