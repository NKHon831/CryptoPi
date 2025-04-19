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
        portfolioManager2 : PortfolioManager = PortfolioManager(),
        broker : BrokerBase = DefaultBroker()
    ):
        self.dataHandler = dataHandler
        self.strategy = strategy 
        self.portfolioManager = portfolioManager
        self.portfolioManager2 = portfolioManager2
        self.end_backtest = end_backtest
        self.broker = broker
        self.currentPortfolioManager = None

    def run(self):
        df_historicalData = self.dataHandler.get_processed_data()
        df_historicalData['trading_signal'] = None 
        
        print("\nRunning backtest...")

        # Iterate through all historical data rows to backtest
        # for datetime, data in df_historicalData.iterrows():
        self.currentPortfolioManager = self.portfolioManager
        for i in range(len(df_historicalData)): 
            data = df_historicalData.iloc[i]
            index = df_historicalData.index[i]

            self.currentPortfolioManager.portfolio.total_signal +=1
            # Trading signal generation and Order creation for current row
            trading_signal = self.strategy.generate_trading_signal(data, index)
            df_historicalData.loc[index, 'trading_signal'] = Signal.map_to_binary(trading_signal)

            if trading_signal in Signal.TRADING_SIGNALS:
                self.currentPortfolioManager.portfolio.total_trading_signal +=1
                self.currentPortfolioManager.generate_order(self.dataHandler.symbol, trading_signal, data)

            # pending orders execution
            if self.currentPortfolioManager.portfolio.get_pending_orders() :
                results = self.broker.execute_orders(self.currentPortfolioManager.send_pending_orders(), self.currentPortfolioManager.portfolio.wallet , data, index)
                self.currentPortfolioManager.update_orders(results)     
                
                # update portfolio 
                previous_data = df_historicalData.iloc[i-1]
                current_data = data
                self.currentPortfolioManager.update_portfolio(previous_data, current_data)

            if (index == self.end_backtest.replace(tzinfo=timezone.utc)):
                # Visualise porfolio stats
                print("Backtest result : ")
                self.currentPortfolioManager.portfolio.overview()
                print("Sharpe ratio", self.currentPortfolioManager.calculate_sharpe_ratio())
                print("Max drawdown: ", self.currentPortfolioManager.get_max_drawdown())
                print("Backtest completed.")
                print()
                print("Running forward test...")

                self.currentPortfolioManager = self.portfolioManager2
           

        print("Forward result : ")
        # Visualise porfolio stats
        self.portfolioManager2.portfolio.overview()
        print("Sharpe ratio", self.portfolioManager2.calculate_sharpe_ratio())
        print("Max drawdown: ", self.portfolioManager2.get_max_drawdown())
        print("Forward test completed.")
        print()
        
        # closed_trades = self.portfolioManager.export_closed_trades()
        # performance_manager = PerformanceManager(closed_trades, self.portfolioManager.portfolio.initial_capital)
        # scalar_metric, time_series_metric = performance_manager.get_metrics()

        # market_data = self.dataHandler.get_processed_data()
        # visualiser = Visualisation(time_series_metric, scalar_metric, market_data)
        # charts = visualiser.plot()
        # # for chart_name, fig in charts.items():
        # #     fig.show()





    


        
            
        
