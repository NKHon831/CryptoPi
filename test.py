from backtesting.datahandler import BaseDataHandler, FinalAlphaModelData
from backtesting.strategy.strategies.sample_strategy import SampleStrategy
from backtesting.strategy.strategies.MovingAverageCrossoverStrategy import MovingAverageCrossoverStrategy
from backtesting.strategy.strategies.BuyAndHoldStrategy import BuyAndHoldStrategy
from backtesting.strategy.strategies.MomentumStrategy import MomentumStrategy
from backtesting.strategy.strategies.AlphaModelStrategy import LogisticModelStrategy
from backtesting.backtest import BackTest
from datetime import datetime

class Main:
    if __name__ == "__main__":

        # Params for fetching the data
        symbol = "BTC"
        start_day = "2020-01-01" # Input the start date in %Y/%m%d format
        end_day = "2023-01-01"
        window = "1h"
        cutoff_date = "2021-12-31" # Last day of backtest_df (Inclusive)

        start_date = datetime.strptime(start_day, "%Y-%m-%d")
        end_date = datetime.strptime(end_day, "%Y-%m-%d")
        cutoff_date = datetime.strptime(cutoff_date, "%Y-%m-%d")
        
        # Initialize the data handler
        BasedataHandler = BaseDataHandler(symbol, start_date, end_date, window)
        AlphaModelDataHandler = FinalAlphaModelData(symbol, start_date, end_date, window)

        # Fetch OHLC data
        df_historical_data = BasedataHandler.get_processed_data()

        # predefined strategies for users
        moving_average_crossover_strategy = MovingAverageCrossoverStrategy(df_historical_data)
        buy_and_hold_strategy = BuyAndHoldStrategy()
        momentum_strategy = MomentumStrategy(df_historical_data)
        sample_strategy = SampleStrategy()

        logistic_model_strategy = LogisticModelStrategy(AlphaModelDataHandler, cutoff_date)

        backtest = BackTest(BasedataHandler, logistic_model_strategy, cutoff_day=cutoff_date)
        backtest.run() 

        # forward test later