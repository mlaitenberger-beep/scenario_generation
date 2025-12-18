class ModelAdapter:
    def __init__(self, config, model):
        self.model_config = None
        self.config = config
        self.model = model 
        self.trainer = None

    def load_model(self):
        trainer = self.model.load_model(self.model_config)
    def train(self):
        self.model.train(self.trainer)
    def predict(self):
        self.model.predict(self)
    def create_model_config(self,config):
        self.model_config =  self.model.create_model_config(self.config)
