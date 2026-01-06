from data.inputData import InputData


class ModelAdapter:
    def __init__(self, config, model):
        self.model_config = None
        self.config = config
        self.model = model 
        self.trainer = None
        self.sample = None
        self.scalar = None

    # Die Input Daten hängen sehr vom model ab#
    #Neben der algemeinen Input funktion braucht es glaube ich noch
    #eine Modell spezifische
    def load_model(self):
        #Wa ist mit bereits trainierten modellen
        self.model.data_input()
        #Muss man nicht hie rnoch eine Model Instanz erstellen 
        #Instanz = Model(self.model_config,self.config, data_hist, data_forcast)
        self.trainer = self.model.load_model()
    def train(self):
        self.model.train(self.trainer)
    def predict(self):
        self.sample = self.model.predict()


    def create_model_config(self,config):
        self.model_config =  self.model.create_model_config(self.config)
    def data_output(self):
        #sample liegt komplet normalisiert vor 
        #rück normalisieren
        #als csv abspeichern 
        pass
