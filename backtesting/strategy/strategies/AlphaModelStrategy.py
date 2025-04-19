from ..StrategyBase import StrategyBase
from ...models.alpha.logistic_regression import AlphaLogisticRegression
from backtesting.constants import Signal

from datetime import datetime, timezone



class LogisticModelStrategy(StrategyBase):
    def __init__(self, dataHandler, cutoff_date=None):
        super().__init__()
        self.dataHandler = dataHandler
        self.cutoff_date = cutoff_date
        self.logistic_model = AlphaLogisticRegression(self.dataHandler, cutoff_date)
        self.backtest_df, self.forward_test_df = self.logistic_model.run_full_pipeline()

    
    def generate_trading_signal(self, data=None, datetime=None):

        print("Generating trading signal...")
        if datetime < self.cutoff_date:
            # Backtest data
            # Fetch the trading signal from the backtest_df
            if datetime not in self.backtest_df["date"]:
                return Signal.HOLD
            trading_signal = self.backtest_df.loc[datetime, 'predictions']
            if trading_signal == 1:
                return Signal.BUY
            elif trading_signal == 0:
                return Signal.Sell
        else:
            if datetime not in self.forward_test_df["date"]:
                return Signal.HOLD
            trading_signal = self.forward_test_df.loc[datetime, 'predictions']
            if trading_signal == 1:
                return Signal.BUY
            elif trading_signal == 0:
                return Signal.Sell

        