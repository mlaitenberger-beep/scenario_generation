import os
from data.inputData import InputData
from adapters import Diffusion_ts_adapter

class ModelAdapter:
    def __init__(self, config, model):
        self.model_config = None
        self.config = config
        self.model = model 
        self.adapter = None
        self.trainer = None
        self.sample = None
        self.scalar = None

        # instantiate model-specific adapter
        if self.model == "diffusion_ts":
            # allow override via config, otherwise use default diffuse config path in repo
            default_cfg = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'diffusion_configs', 'diffusion_config.yaml'))
            diffusion_cfg_path = self.config.get('diffusion_config_path', default_cfg)
            self.adapter = Diffusion_ts_adapter(diffusion_cfg_path, self.config)
        
    def load_model(self):
        #Wa ist mit bereits trainierten modellen
        # prepare data and create trainer/model
        self.adapter.data_input()
        self.trainer = self.adapter.load_model()
    def train(self):
        # train or resume using trainer; adapter may return an updated trainer
        self.trainer = self.adapter.train(self.trainer)
    def predict(self):
        # pass trainer into predict
        self.sample = self.adapter.predict(self.trainer)


    def create_model_config(self,config=None):
        # allow passing an override config dict; merge into existing config
        if config is not None:
            self.config.update(config)
        self.model_config = self.adapter.create_model_config()
        return self.model_config
    def data_output(self):
        #sample liegt komplet normalisiert vor 
        #rück normalisieren
        #als csv abspeichern 
        pass
