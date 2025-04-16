from ..StrategyBase import StrategyBase
from ...models.alpha.test import FinalAlphaModel

class SampleStrategy(StrategyBase):
    def __init__(self):
        super().__init__()
        self.finalAlphaModel = FinalAlphaModel()

    def generate_trading_signal(self, data):
        predicted_signal = self.finalAlphaModel.predict()
        if predicted_signal % 2 == 0 :
            return self.buy()
        else:
            return self.sell()
        