from backtesting.portfolio import Portfolio
from backtesting.datahandler import BaseDataHandler
from backtesting.strategy.StrategyBase import StrategyBase
from backtesting.constants import Signal
from backtesting.execution.Order import Order
from backtesting.execution.broker.brokers.DefaultBroker import DefaultBroker
from backtesting.execution.broker.BrokerBase import BrokerBase

class BackTest:
    def __init__(
        self, 
        dataHandler : BaseDataHandler, 
        strategy : StrategyBase, 
        portfolio : Portfolio = Portfolio(),
        broker : BrokerBase = DefaultBroker()
    ):
        self.dataHandler = dataHandler
        self.strategy = strategy 
        self.portfolio = portfolio   
        self.broker = broker

    def run(self):
        print("\nRunning Backtest...")
        df_historicalData = self.dataHandler.get_data()

        # print(df_historicalData.head())

        # Iterate through all historical data rows to backtest
        for index, row in df_historicalData.iterrows():

            # pending orders execution
            results = self.broker.execute_orders(self.portfolio.send_pending_orders())
            self.portfolio.update_records(results)

            # Trading signal generation and Order creation for current row
            trading_signal = self.strategy.generate_trading_signal(row)
            if trading_signal in Signal.TRADING_SIGNALS:
                new_order = Order.create(row)
                self.portfolio.add_pending_order(new_order)

    def get_portfolio(self):
        return self.portfolio

