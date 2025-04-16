import random

class Model:
    def __init__(self, model_type, model):
        self.model_type = model_type
        self.model = model

    def predict(self, data):
        if self.model_type == 'sklearn':
            return self.model.predict(data)
        elif self.model_type == 'keras':
            return self.model.predict(data)
        else:
            raise ValueError("Unsupported model type")
        

class FinalAlphaModel(Model):
    def __init__(self):
        pass


    def predict(self):
        return 'buy' if random.randint(0,10) % 2 == 0 else 'sell'