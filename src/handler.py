

class Handler:
    def __init__(self, modelAdapter,config):
        self.modelAdapter = modelAdapter
        self.config = config

    def createScenarios(self, modelAdapter, config):
        modelAdapter.create_model_config(config)
        modelAdapter.load_model()
        modelAdapter.train()
        modelAdapter.predict()

    #data output
        #Wie sollen die Daten ausgegeben werden? Als Pfad, als Dataframe, als np array? 
