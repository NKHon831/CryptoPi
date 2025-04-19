from backtesting.portfolio.portfolio import Portfolio
from backtesting.datahandler import BaseDataHandler
from backtesting.strategy.StrategyBase import StrategyBase
from backtesting.constants import Signal, MarketEntryType
from backtesting.execution.Order import Order
from backtesting.execution.broker.brokers.DefaultBroker import DefaultBroker
from backtesting.execution.broker.BrokerBase import BrokerBase
from backtesting.portfolio.portfolio_manager import PortfolioManager
from backtesting.performance.performance_base import PerformanceBase
from backtesting.performance.performance_manager import PerformanceManager
from backtesting.visualisation.visualisation import StrategyVisualisation

class BackTest:
    def __init__(
        self, 
        dataHandler : BaseDataHandler, 
        strategy : StrategyBase, 
        # portfolio : Portfolio = Portfolio(),
        performance : PerformanceBase = PerformanceBase(),
        portfolioManager : PortfolioManager = PortfolioManager(),
        broker : BrokerBase = DefaultBroker()
    ):
        self.dataHandler = dataHandler
        self.strategy = strategy 
        self.portfolioManager = portfolioManager   
        self.broker = broker
        self.performance = performance

    def run(self):
        print("\nRunning backtest...")
        df_historicalData = self.dataHandler.get_processed_data()
        df_historicalData['trading_signal'] = None 

        # Iterate through all historical data rows to backtest
        for datetime, data in df_historicalData.iterrows():

            # Trading signal generation and Order creation for current row
            trading_signal = self.strategy.generate_trading_signal(data, datetime)
            df_historicalData.loc[datetime, 'trading_signal'] = Signal.map_to_binary(trading_signal)

            if trading_signal in Signal.TRADING_SIGNALS:
                self.portfolioManager.generate_order(self.dataHandler.symbol, trading_signal, data)

            # pending orders execution
            if self.portfolioManager.portfolio.get_pending_orders() :
                results = self.broker.execute_orders(self.portfolioManager.send_pending_orders(), self.portfolioManager.portfolio.wallet , data, datetime)
                self.portfolioManager.update_orders(results)

        print("\nBacktest completed.")

        closed_trades = self.portfolioManager.export_closed_trades()
        performance_manager = PerformanceManager(closed_trades, self.portfolioManager.portfolio.initial_capital)
        scalar_metric, time_series_metric = performance_manager.get_metrics()

        # # Pass closed_trades to performance
        # PerformanceBase.import_closed_trades(self.portfolioManager.export_closed_trades())

        # # Pass market data with signal for visualisation
        # StrategyVisualisation.import_market_data_with_trading_signal(self.dataHandler.get_processed_data())

        # Visualise porfolio stats
        print("\nPortfolio Overview:")
        self.portfolioManager.portfolio.overview()
        
            
        
