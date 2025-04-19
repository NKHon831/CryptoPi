from backtesting.strategy.StrategyBase import StrategyBase
from backtesting.constants import Signal

class BuyAndHoldStrategy(StrategyBase):

    def __init__(self):
        self.buy = True

    def generate_trading_signal(self, data, datetime):
        if self.buy:
            self.buy = False
            return Signal.BUY
        else :
            return Signal.HOLD

