
from backtesting.strategy.StrategyBase import StrategyBase
from backtesting.constants import Signal

class MomentumStrategy(StrategyBase):
    def __init__(self, df_historical_data , window = 20):
        self.df = df_historical_data
        self.df['momentum'] = self.df['close']['BTC'].pct_change(periods = window).mean()
        pass

    def generate_trading_signal(self, data, datetime):
        momentum = self.df['momentum'].get(datetime, None)
        if(momentum is None or momentum == 0):
            return Signal.HOLD
        elif(momentum > 0):
            return Signal.BUY
        elif(momentum < 0):
            return Signal.SELL