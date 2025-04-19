import pandas as pd

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
class BackTest:
    def __init__(
        self, 
        dataHandler : BaseDataHandler, 
        strategy : StrategyBase, 
        # portfolio : Portfolio = Portfolio(),
        portfolioManager : PortfolioManager = PortfolioManager(),
        broker : BrokerBase = DefaultBroker(),
        cutoff_day = None,
    ):
        self.dataHandler = dataHandler
        self.strategy = strategy 
        self.portfolioManager = portfolioManager   
        self.broker = broker
        self.cutoff_day = cutoff_day
        

    def run(self):
        print("\nRunning backtest...")
        df_historicalData = self.dataHandler.get_processed_data()
        df_historicalData['trading_signal'] = None 



        # Get all the predictions from the model
        # backtest_df, forward_test_df = self.strategy.generate_trading_signal()
        # print("BacktestData\n")
        # print(backtest_df.head(5))
        # # If 'date' in backtest_df is not already datetime, convert it
        # backtest_df['date'] = pd.to_datetime(backtest_df['date'], utc=True)

        # df_historicalData = df_historicalData.reset_index()

        # # Rename the datetime index column (probably named something like "index" or "Datetime") to "date"
        # df_historicalData = df_historicalData.rename(columns={'timestamp': 'date'})

        # df_historicalData['date'] = pd.to_datetime(df_historicalData['date'])
        # if df_historicalData['date'].dt.tz is None:
        #     df_historicalData['date'] = df_historicalData['date'].dt.tz_localize(None)

        # # Make sure both are datetime dtype
        # # df_historicalData['date'] = pd.to_datetime(df_historicalData['date'])
        # print("Historical Data\n")
        # print(df_historicalData.head(5))

        # # Perform the inner join on 'date'
        # merged_df = pd.merge(df_historicalData, backtest_df, on='date', how='inner')

        # # Sort by date just in case
        # merged_df = merged_df.sort_values(by='date')

        # merged_df['trading_signal'] = merged_df['predictions']
        # merged_df.drop(columns=['predictions'], inplace=True)

        # print(merged_df.head(5))

        # # Iterate through all historical data rows to backtest
        # for datetime, data in merged_df.iterrows():

        #     # Trading signal generation and Order creation for current row
        #     trading_signal = merged_df["trading_signal"].loc[datetime]
        #     if trading_signal == 1:
        #         trading_signal = Signal.BUY
        #     elif trading_signal == 0:
        #         trading_signal = Signal.SELL
        #     # df_historicalData.loc[datetime, 'predictions'] = Signal.map_to_binary(trading_signal)

        #     print("Trading Signal: ", trading_signal)
        #     if trading_signal in Signal.TRADING_SIGNALS:
        #         self.portfolioManager.generate_order(self.dataHandler.symbol, trading_signal, data)

        #     # pending orders execution
        #     if self.portfolioManager.portfolio.get_pending_orders() :
        #         results = self.broker.execute_orders(self.portfolioManager.send_pending_orders(), self.portfolioManager.portfolio.wallet , data, data["date"])
        #         self.portfolioManager.update_orders(results)

        # print("\nBacktest completed.")


        # Iterate through all historical data rows to backtest
        for dt, data in df_historicalData.iterrows():

            # print("Datetime: ", dt)
            current_datetime = dt.replace(tzinfo=None) # Convert to naive datetime
            print("Naive Datetime: ", current_datetime)

            # Trading signal generation and Order creation for current row
            trading_signal = self.strategy.generate_trading_signal(data, current_datetime)
            print("Trading Signal: ", trading_signal)
            df_historicalData.loc[current_datetime, 'trading_signal'] = Signal.map_to_binary(trading_signal)

            if trading_signal in Signal.TRADING_SIGNALS:
                print("Generating order...")
                self.portfolioManager.generate_order(self.dataHandler.symbol, trading_signal, data)

            # pending orders execution
            if self.portfolioManager.portfolio.get_pending_orders() :
                print("Executing orders...")
                results = self.broker.execute_orders(self.portfolioManager.send_pending_orders(), self.portfolioManager.portfolio.wallet , data, current_datetime)
                self.portfolioManager.update_orders(results)

        print("\nBacktest completed.")

        closed_trades = self.portfolioManager.export_closed_trades()
        performance_manager = PerformanceManager(closed_trades, self.portfolioManager.portfolio.initial_capital)
        scalar_metric, time_series_metric = performance_manager.get_metrics()

        market_data = self.dataHandler.get_processed_data()
        visualiser = Visualisation(time_series_metric, scalar_metric, market_data)
        charts = visualiser.plot()
        for chart_name, fig in charts.items():
            fig.show()

        # Visualise porfolio stats
        print("\nPortfolio Overview:")
        self.portfolioManager.portfolio.overview()
        
            
        
