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
from backtesting.visualisation.visualisation import Visualisation
from backtesting.datahandler import BenchmarkData
from datetime import datetime

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
        df_historicalData = self.dataHandler.get_processed_data()
        df_historicalData['trading_signal'] = None 

        # Iterate through all historical data rows to backtest
        for dt, data in df_historicalData.iterrows():

            # Trading signal generation and Order creation for current row
            trading_signal = self.strategy.generate_trading_signal(data, dt)
            df_historicalData.loc[dt, 'trading_signal'] = Signal.map_to_binary(trading_signal)

            if trading_signal in Signal.TRADING_SIGNALS:
                self.portfolioManager.generate_order(self.dataHandler.symbol, trading_signal, data)

            # pending orders execution
            if self.portfolioManager.portfolio.get_pending_orders() :
                results = self.broker.execute_orders(self.portfolioManager.send_pending_orders(), self.portfolioManager.portfolio.wallet , data, dt)
                self.portfolioManager.update_orders(results)

        print("\nBacktest completed.")


        start = datetime(2022, 1, 1)
        end = datetime(2022, 12, 31)
        benchmark_df = BenchmarkData("BTC-USD", start, end).fetch_yfinance_data()
        benchmark_curve = benchmark_df.set_index('Date')['benchmark']
        print(benchmark_curve)

        closed_trades = self.portfolioManager.export_closed_trades()
        performance_manager = PerformanceManager(closed_trades, self.portfolioManager.portfolio.initial_capital, benchmark_curve)
        scalar_metric, time_series_metric = performance_manager.get_metrics()
        print(scalar_metric)

        market_data = self.dataHandler.get_processed_data()
        visualiser = Visualisation(time_series_metric, scalar_metric, market_data)
        charts = visualiser.plot()
        for chart_name, fig in charts.items():
            fig.show()

        # Visualise porfolio stats
        print("\nPortfolio Overview:")
        self.portfolioManager.portfolio.overview()
        
            
        
