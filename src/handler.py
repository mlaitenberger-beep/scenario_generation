from adapters.modelAdapter import ModelAdapter

class Handler:
    def __init__(self,model,config):
        self.modelAdapter = None
        self.config = config
        self.modelAdapter = ModelAdapter(self.config, model)

    def createScenarios(self, config=None):
        """High-level flow: create model config, load data & model, train (or resume), and predict samples.

        Returns the generated samples (numpy array) from the adapter.
        """
        self.modelAdapter.create_model_config(config)
        self.modelAdapter.load_model()
        self.modelAdapter.train()
        self.modelAdapter.predict()
        # denormalize back to original scale
        return self.modelAdapter.data_output()
