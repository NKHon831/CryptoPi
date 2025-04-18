from ..StrategyBase import StrategyBase
from ...models.alpha.test import FinalAlphaModel
from backtesting.constants import Signal
import random


class SampleStrategy(StrategyBase):
    def __init__(self):
        super().__init__()
        self.finalAlphaModel = FinalAlphaModel()

    def generate_trading_signal(self, data, datetime = None):
        # if(data['high']['BTC'] > 43.2):
        if(random.random() <= 0.5):
            return Signal.BUY
        else :
            return Signal.SELL

        

        