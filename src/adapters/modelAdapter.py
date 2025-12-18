class ModelAdapter:
    def __init__(self, config, model):
        self.model_config = None
        self.config = config
        self.model = model 
        self.trainer = None
        self.sample = None
    def data_input(self):
        pass
    def data_output(self):
        pass
    def load_model(self):
        data_hist, data_forcast = data_input()
        trainer = self.model.load_model(self.model_config, data_hist, data_forcast)
    def train(self):
        self.model.train(self.trainer)
    def predict(self):
        self.sample = self.model.predict(self)
    def create_model_config(self,config):
        self.model_config =  self.model.create_model_config(self.config)
    

