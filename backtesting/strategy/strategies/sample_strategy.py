from ..StrategyBase import StrategyBase
from ...models.alpha.test import FinalAlphaModel
from backtesting.constants import Signal
import random


class SampleStrategy(StrategyBase):
    total_buy = 0
    total_sell = 0
    def __init__(self):
        super().__init__()
        self.finalAlphaModel = FinalAlphaModel()

    def generate_trading_signal(self, data):
        # if(data['high']['BTC'] > 43.2):
        if(random.random() <= 0.5):
            SampleStrategy.total_buy +=1
            return Signal.BUY
        else :
            SampleStrategy.total_sell +=1
            return Signal.SELL

        

        