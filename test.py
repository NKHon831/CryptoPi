from backtesting.datahandler import BaseDataHandler
from backtesting.strategy.strategies.sample_strategy import SampleStrategy
from backtesting.strategy.strategies.MovingAverageCrossoverStrategy import MovingAverageCrossoverStrategy
from backtesting.strategy.strategies.BuyAndHoldStrategy import BuyAndHoldStrategy
from backtesting.strategy.strategies.MomentumStrategy import MomentumStrategy
from backtesting.strategy.strategies.MarketRegimeHMM import MarketRegimeHMM
from backtesting.backtest import BackTest
from datetime import datetime

class Main:
    if __name__ == "__main__":
        dataHandler = BaseDataHandler(symbol='BTC',
                          start_time=datetime(2025, 1, 1),
                          end_time=datetime(2025, 4, 15),
                          window="1h")
        
        
        # predefined strategies for users
        moving_average_crossover_strategy = MovingAverageCrossoverStrategy(dataHandler.get_processed_data())
        buy_and_hold_strategy = BuyAndHoldStrategy()
        momentum_strategy = MomentumStrategy(dataHandler.raw_data)
        ml_based_strategy = MarketRegimeHMM()

        backtest = BackTest(dataHandler, ml_based_strategy, datetime(2025, 3, 1))
        backtest.run()

        #forward test later
