from backtesting.datahandler import BaseDataHandler
from backtesting.strategy.strategies.sample_strategy import SampleStrategy
from backtesting.backtest import BackTest
from datetime import datetime

class Main:
    if __name__ == "__main__":
        dataHandler = BaseDataHandler(symbol='BTC',
                          start_time=datetime(2025, 1, 1),
                          end_time=datetime(2025, 4, 15),
                          window="1h")
        strategy = SampleStrategy()
        backtest = BackTest(dataHandler, strategy)
        backtest.run() 
        print("Backtest completed.\n")
        
