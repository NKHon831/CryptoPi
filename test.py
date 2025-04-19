from backtesting.datahandler import BaseDataHandler
from backtesting.strategy.strategies.sample_strategy import SampleStrategy
from backtesting.strategy.strategies.MovingAverageCrossoverStrategy import MovingAverageCrossoverStrategy
from backtesting.strategy.strategies.BuyAndHoldStrategy import BuyAndHoldStrategy
from backtesting.strategy.strategies.MomentumStrategy import MomentumStrategy
from backtesting.backtest import BackTest
from datetime import datetime

class Main:
    if __name__ == "__main__":
        dataHandler = BaseDataHandler(symbol='BTC',
                          start_time=datetime(2025, 1, 1),
                          end_time=datetime(2025, 4, 15),
                          window="1h")
        
        df_historical_data = dataHandler.get_processed_data()

        # predefined strategies for users
        moving_average_crossover_strategy = MovingAverageCrossoverStrategy(df_historical_data)
        buy_and_hold_strategy = BuyAndHoldStrategy()
        momentum_strategy = MomentumStrategy(df_historical_data)
        

        sample_strategy = SampleStrategy()
        backtest = BackTest(dataHandler, moving_average_crossover_strategy)
        backtest.run() 

        #forward test later
