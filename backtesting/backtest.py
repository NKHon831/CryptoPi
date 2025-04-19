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
class BackTest:
    
    def __init__(
        self, 
        dataHandler : BaseDataHandler, 
        strategy : StrategyBase, 
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
        # for datetime, data in df_historicalData.iterrows():
        for i in range(len(df_historicalData)):
            data = df_historicalData.iloc[i]
            datetime = df_historicalData.index[i]

            # Trading signal generation and Order creation for current row
            trading_signal = self.strategy.generate_trading_signal(data, datetime)
            df_historicalData.loc[datetime, 'trading_signal'] = Signal.map_to_binary(trading_signal)

            if trading_signal in Signal.TRADING_SIGNALS:
                self.portfolioManager.generate_order(self.dataHandler.symbol, trading_signal, data)


            num_of_executed_order = 0
            # pending orders execution
            if self.portfolioManager.portfolio.get_pending_orders() :
                results = self.broker.execute_orders(self.portfolioManager.send_pending_orders(), self.portfolioManager.portfolio.wallet , data, datetime)
                self.portfolioManager.update_orders(results)

                num_of_executed_order = len(results['executed_orders'])
            
            executed_order = None
            # Update position by row
            if(len(self.portfolioManager.portfolio.executed_orders) != 0):
                executed_order = self.portfolioManager.portfolio.executed_orders[-1]

            if(len(self.portfolioManager.portfolio.positions) == 0):
                position = Position(Direction.NEUTRAL,0,0,0,0)
                self.portfolioManager.portfolio.positions.append(position)

            else:
                previous_position = self.portfolioManager.portfolio.positions[-1]

                updated_holdings = 0
                if(executed_order is None):
                    updated_holdings = previous_position.size
                else:
                    if(executed_order.trading_signal is Signal.BUY):
                        updated_holdings = previous_position.size + executed_order.quantity
                    else:
                        updated_holdings = previous_position.size - executed_order.quantity


                direction = 0
                if(updated_holdings > 0):
                    direction = 1
                elif(updated_holdings < 0):
                    direction = -1
                else:
                    direction = 0
                
                new_position = Position(direction, updated_holdings)

                #calculate pnl
                previous_row = df_historicalData.iloc[i-1]
                price_change = (data['close'] / previous_row['close']) -1

                pnl = (price_change * previous_position.direction * previous_position.size) - (0.0006 * num_of_executed_order)
                new_position.pnl = pnl
                equity = sum(position.pnl for position in self.portfolioManager.portfolio.positions)
                new_position.equity = equity

                self.portfolioManager.portfolio.positions.append(new_position)
                
        print("\nBacktest completed.")

        # closed_trades = self.portfolioManager.export_closed_trades()
        # performance_manager = PerformanceManager(closed_trades, self.portfolioManager.portfolio.initial_capital)
        # scalar_metric, time_series_metric = performance_manager.get_metrics()

        # market_data = self.dataHandler.get_processed_data()
        # visualiser = Visualisation(time_series_metric, scalar_metric, market_data)
        # charts = visualiser.plot()
        # # for chart_name, fig in charts.items():
        # #     fig.show()

        # Visualise porfolio stats
        print("\nPortfolio Overview:")
        self.portfolioManager.portfolio.overview()

        
            
        
