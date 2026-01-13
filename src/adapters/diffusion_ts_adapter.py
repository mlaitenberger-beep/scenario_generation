import os
import numpy as np
import sys
import pandas as pd
from data.inputData import InputData

# Delay heavy/third-party imports until methods are called. Importing torch
# or the Diffusion-TS engine at module import time can be slow or may trigger
# device/CUDA initialization that blocks test collection; import lazily instead.

def _lazy_imports():
    """Return a tuple of (torch, Trainer, build_dataloader, load_yaml_config, instantiate_from_config).
    Raises ImportError with an informative message if something is missing."""
    # make sure Diffusion-TS root is on sys.path so its internal absolute imports resolve
    diffusion_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'DiffusionTS'))
    if diffusion_root not in sys.path:
        sys.path.insert(0, diffusion_root)

    try:
        import torch
    except Exception as e:
        raise ImportError("Failed to import torch; ensure torch is installed and importable") from e

    try:
        from engine.solver import Trainer
        from Data.build_dataloader import build_dataloader
        from Utils.io_utils import load_yaml_config, instantiate_from_config
    except Exception as e:
        raise ImportError("Failed to import Diffusion-TS internals. Ensure the third_party/DiffusionTS package is present and its dependencies are installed.") from e

    return torch, Trainer, build_dataloader, load_yaml_config, instantiate_from_config
from .modelAdapter import ModelAdapter
from data.utils import CustomDataset, Args_Example

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
        # load yaml via lazy imports so this method doesn't require Diffusion-TS
        # internals to be present at import time
        _, _, _, load_yaml_config, _ = _lazy_imports()
        cfg = load_yaml_config(self.diffusionTS_config_path)

        # Ensure expected nested dicts exist
        model_params = cfg.setdefault('model', {}).setdefault('params', {})
        data_params = cfg.setdefault('data', {}).setdefault('params', {})
        dataloader = cfg.setdefault('dataloader', {})
        train_ds_params = dataloader.setdefault('train_dataset', {}).setdefault('params', {})
        test_ds_params = dataloader.setdefault('test_dataset', {}).setdefault('params', {})
        solver_cfg = cfg.setdefault('solver', {})

        # set model and data related params from the higher-level config
        seq = self.config.get('seq_length') or self.config.get('sequence_length')
        if seq is not None:
            model_params['seq_length'] = seq
            model_params['sequence_length'] = seq

        feat = self.config.get('feature_size')
        if feat is not None:
            model_params['feature_size'] = feat
            data_params['feature_size'] = feat

        results_folder = self.config.get('results_folder')
        if results_folder:
            solver_cfg['results_folder'] = results_folder

        data_root = self.config.get('data_root')
        if data_root:
            train_ds_params['data_root'] = data_root
            test_ds_params['data_root'] = data_root

        return cfg
    

    def load_model(self):
        # build dataloader from prepared historical sequences
        # build dataloader from prepared historical sequences
        hist_dataset = CustomDataset(data=self.data_hist)

        # import torch and Diffusion-TS internals lazily
        torch, Trainer, build_dataloader, load_yaml_config, instantiate_from_config = _lazy_imports()

        DataLoader = torch.utils.data.DataLoader
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
        # use lazy imports so prediction can be called even in test environments without torch
        torch, Trainer, build_dataloader, load_yaml_config, instantiate_from_config = _lazy_imports()

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

    def data_output(self, samples):
        """Denormalize model outputs back to the original (relative-change) scale.

        Steps:
        - Convert from [-1, 1] back to [0, 1]
        - Inverse transform via the fitted MinMaxScaler (fit on historical relative changes)
        - Optionally save aggregated absolute values to a single CSV when
          `output_csv_path` is provided in the config.

        Returns a numpy array with the same shape as `samples` (relative changes).
        """
        if samples is None:
            return None
        if self.scalar is None:
            raise ValueError("Scaler not available. Ensure data_input() ran successfully before data_output().")

        feat = self.config.get('feature_size')
        if feat is None:
            raise ValueError("feature_size missing in config; required for reshaping outputs during denormalization.")

        def _unnormalize_to_zero_to_one(x):
            return (x + 1.0) / 2.0

        flat = samples.reshape(-1, feat)
        flat_01 = _unnormalize_to_zero_to_one(flat)

        denorm_flat = self.scalar.inverse_transform(flat_01)
        denorm = denorm_flat.reshape(samples.shape)

        # Optionally write all samples as absolute values into one CSV for downstream tools
        output_csv_path = self.config.get('output_csv_path')
        if output_csv_path:
            hist_path = self.config.get('data_hist_path')
            if hist_path is None:
                raise ValueError("output_csv_path is set but data_hist_path is missing; required to reconstruct absolute values.")

            hist_df = pd.read_csv(hist_path, header=0, dtype=float)
            last_hist_values = hist_df.iloc[-1].to_numpy()[:feat]

            samples_abs = np.zeros_like(denorm)
            for i in range(denorm.shape[0]):
                current = last_hist_values.copy()
                for t in range(denorm.shape[1]):
                    current = current * (1.0 + denorm[i, t, :])
                    samples_abs[i, t, :] = current

            # Long format: sample_id, time_idx, feature columns
            seq_len = samples_abs.shape[1]
            records = []
            for sample_id in range(samples_abs.shape[0]):
                for t in range(seq_len):
                    records.append([sample_id, t, *samples_abs[sample_id, t, :]])

            headers = ['sample_id', 'time_idx'] + list(hist_df.columns[:feat])
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            pd.DataFrame(records, columns=headers).to_csv(output_csv_path, index=False)
            print(f"Saved aggregated absolute samples to {output_csv_path}")

        return denorm

