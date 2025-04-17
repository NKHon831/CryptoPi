from backtesting.portfolio import Portfolio
from backtesting.datahandler import BaseDataHandler
from backtesting.strategy.StrategyBase import StrategyBase
from backtesting.constants import Signal
from backtesting.execution.Order import Order

class BackTest:
    def __init__(
        self, 
        dataHandler : BaseDataHandler, 
        strategy : StrategyBase, 
        portfolio : Portfolio = Portfolio()
    ):
        self.dataHandler = dataHandler
        self.strategy = strategy 
        self.portfolio = portfolio   

    def run(self):
        print("\nRunning Backtest...")
        totalSell = 0
        totalBuy = 0
        df_historicalData = self.dataHandler.get_data()

        # print(df_historicalData.head())

        # Iterate through all historical data rows to backtest
        for index, row in df_historicalData.iterrows():

            trading_signal = self.strategy.generate_trading_signal(row)
            if trading_signal in Signal.TRADING_SIGNALS:
                new_order = Order.create(row)
                self.portfolio.add_pending_order(new_order)

    def get_portfolio(self):
        return self.portfolio

