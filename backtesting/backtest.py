from backtesting.portfolio.portfolio import Portfolio
from backtesting.datahandler import BaseDataHandler
from backtesting.strategy.StrategyBase import StrategyBase
from backtesting.constants import Signal, Direction
from backtesting.execution.Order import Order
from backtesting.execution.broker.brokers.DefaultBroker import DefaultBroker
from backtesting.execution.broker.BrokerBase import BrokerBase
from backtesting.portfolio.portfolio_manager import PortfolioManager
from backtesting.performance.performance_base import PerformanceBase
from backtesting.performance.performance_manager import PerformanceManager
from backtesting.visualisation.visualisation import Visualisation
from backtesting.portfolio.position import Position
from datetime import datetime, timezone
class BackTest:
    
    def __init__(
        self, 
        dataHandler : BaseDataHandler, 
        strategy : StrategyBase, 
        end_backtest : datetime,
        portfolioManager : PortfolioManager = PortfolioManager(),
        broker : BrokerBase = DefaultBroker()
    ):
        self.dataHandler = dataHandler
        self.strategy = strategy 
        self.portfolioManager = portfolioManager  
        self.end_backtest = end_backtest
        self.broker = broker

    def run(self):
        df_historicalData = self.dataHandler.get_processed_data()
        df_historicalData['trading_signal'] = None 
        
        print("\nRunning backtest...")

        # Iterate through all historical data rows to backtest
        # for datetime, data in df_historicalData.iterrows():
        for i in range(len(df_historicalData)): 
            data = df_historicalData.iloc[i]
            index = df_historicalData.index[i]
            
            # Trading signal generation and Order creation for current row
            trading_signal = self.strategy.generate_trading_signal(data, index)
            df_historicalData.loc[index, 'trading_signal'] = Signal.map_to_binary(trading_signal)

            if trading_signal in Signal.TRADING_SIGNALS:
                self.portfolioManager.generate_order(self.dataHandler.symbol, trading_signal, data)

            # pending orders execution
            if self.portfolioManager.portfolio.get_pending_orders() :
                results = self.broker.execute_orders(self.portfolioManager.send_pending_orders(), self.portfolioManager.portfolio.wallet , data, index)
                self.portfolioManager.update_orders(results)     
                
                # update portfolio 
                previous_data = df_historicalData.iloc[i-1]
                current_data = data
                self.portfolioManager.update_portfolio(previous_data, current_data)
                   
        print("Backtest completed.")
        print("Backtest result : ")

        # Visualise porfolio stats
        print("\nPortfolio Overview:")
        self.portfolioManager.portfolio.overview()

        print("Max drawdown: ", self.portfolioManager.get_max_drawdown())
        print()
        
        # closed_trades = self.portfolioManager.export_closed_trades()
        # performance_manager = PerformanceManager(closed_trades, self.portfolioManager.portfolio.initial_capital)
        # scalar_metric, time_series_metric = performance_manager.get_metrics()

        # market_data = self.dataHandler.get_processed_data()
        # visualiser = Visualisation(time_series_metric, scalar_metric, market_data)
        # charts = visualiser.plot()
        # # for chart_name, fig in charts.items():
        # #     fig.show()





    


        
            
        
