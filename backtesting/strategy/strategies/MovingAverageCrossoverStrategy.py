from backtesting.strategy.StrategyBase import StrategyBase
from backtesting.constants import Signal

class MovingAverageCrossoverStrategy(StrategyBase):

    def __init__(self, df_historical_data , short_window = 20, long_window = 50):
        self.df = df_historical_data
        self.df[('SMA' , 'BTC')] = self.df['close']['BTC'].rolling(window = short_window).mean()
        self.df[('LMA' , 'BTC')] = self.df['close']['BTC'].rolling(window = long_window).mean()
        pass

    def generate_trading_signal(self, data, datetime):
        sma = self.df['SMA']['BTC'].loc[datetime]
        lma = self.df['LMA']['BTC'].loc[datetime]

        if sma is None or lma is None or sma == lma:
            return Signal.HOLD 
    
        if(sma > lma) :
            return Signal.BUY
        elif(lma > sma):
            return Signal.SELL