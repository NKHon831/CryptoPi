from backtesting.portfolio.portfolio import Portfolio
from backtesting.datahandler import BaseDataHandler
from backtesting.strategy.StrategyBase import StrategyBase
from backtesting.constants import Signal, MarketEntryType
from backtesting.execution.Order import Order
from backtesting.execution.broker.brokers.DefaultBroker import DefaultBroker
from backtesting.execution.broker.BrokerBase import BrokerBase
from backtesting.portfolio.portfolio_manager import PortfolioManager

class BackTest:
    def __init__(
        self, 
        dataHandler : BaseDataHandler, 
        strategy : StrategyBase, 
        # portfolio : Portfolio = Portfolio(),
        portfolioManager : PortfolioManager = PortfolioManager(),
        broker : BrokerBase = DefaultBroker()
    ):
        self.dataHandler = dataHandler
        self.strategy = strategy 
        self.portfolioManager = portfolioManager   
        self.broker = broker

    def run(self):
        print("\nRunning backtest...")
        df_historicalData = self.dataHandler.get_data()

        # Iterate through all historical data rows to backtest
        for datetime, data in df_historicalData.iterrows():

            # pending orders execution
            if self.portfolioManager.portfolio.get_pending_orders() :
                results = self.broker.execute_orders(self.portfolioManager.send_pending_orders(), self.portfolioManager.portfolio.wallet , data, datetime)
                self.portfolioManager.update_orders(results)

            # Trading signal generation and Order creation for current row
            trading_signal = self.strategy.generate_trading_signal(data)
            if trading_signal in Signal.TRADING_SIGNALS:
                self.portfolioManager.generate_order(trading_signal, data)

        print("\nBacktest completed.")

        # print("Trade history\n")
        # for trade in self.portfolioManager.portfolio.get_all_trades():
        #     print(trade)

        # Visualise porfolio stats
        print("\nPortfolio Overview:")
        self.portfolioManager.portfolio.overview()
