from ..StrategyBase import StrategyBase
from ...models.alpha.test import FinalAlphaModel
from ...constants import Signal

class SampleStrategy(StrategyBase):
    def __init__(self):
        super().__init__()
        self.finalAlphaModel = FinalAlphaModel()

    def generate_trading_signal(self, data):
        if(data['high']['BTC'] > 43.2):
            return Signal.BUY
        else :
            return Signal.SELL

        