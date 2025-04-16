from backtesting.datahandler import DataHandler
from backtesting.strategy.strategies.sample_strategy import SampleStrategy
from backtesting.backtest import BackTest

class Main:
    if __name__ == "__main__":
        dataHandler = DataHandler(data=[1,2,3,4,5])
        strategy = SampleStrategy()

        backtest = BackTest(dataHandler, strategy)

        backtest.run()