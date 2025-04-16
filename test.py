from backtesting.datahandler import DataHandler
from backtesting.strategy import SampleStrategy
from backtesting.backtest import Backtest

class Main:
    if __name__ == "__main__":
        dataHandler = DataHandler([1,2,3,4,5])
        strategy  = SampleStrategy()

        backtest = Backtest(strategy, dataHandler)

        backtest.run()


