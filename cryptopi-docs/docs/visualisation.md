## MarketVisualisation

`MarketVisualisation` is a class designed to generate interactive charts using Plotly for visualizing market data, specifically price charts.

### Parameters

| Parameter   | Type         | Description                                                                                                                                        |
| ----------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| market_data | pd.DataFrame | A DataFrame containing market data with columns such as 'datetime', 'open', 'high', 'low', and 'close'. This data will be used to plot the charts. |

### Methods

`plot_price_chart(self, title='BTC-USD Price Chart')` returns a Plotly Figure object representing the price candlestick chart.

## StrategyVisualisation

`StrategyVisualisation` is an extension of MarketVisualisation that includes additional functionality for visualizing the performance of a trading strategy, including equity curves, daily returns, performance metrics, and price charts with signals.

### Methods

| Method                                                                                  | Description                                                                                                    |
| --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| plot_equity_curve(self, title="Equity Curve")                                           | Plots the equity curve for each strategy in the performance_data.                                              |
| tabulate_metrics(self, title="Performance Metrics Summary")                             | Creates a table displaying the performance metrics for the strategy.                                           |
| plot_daily_returns(self, title="Daily Returns")                                         | Plots the daily returns for each strategy in the performance_data.                                             |
| plot_price_with_signals(self, title="Price Chart with Signals", simple_plot_only=False) | Plots the price chart along with buy and sell signals.                                                         |
| plot(self, all=True, charts=None)                                                       | Plots the requested charts either all at once or a specific set of charts.                                     |
| export(self, format="html", all=True, charts=None, export_dir="exports")                | Exports the charts in the specified format to the specified directory.                                         |
| load_market_data(self, market_data)                                                     | Loads the market data into the market_data attribute and updates the chart_mapping to include the price chart. |

## ComparisonPlot

The `ComparisonPlot` class provides a suite of interactive charts using Plotly for comparing the performance of multiple backtesting strategies.

### Parameters

| Parameter            | Type                        | Description                                                                                                             |
| -------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| performances_data    | dict[str, pd.DataFrame]     | Dictionary where keys are strategy names and values are DataFrames with columns datetimes, equity_values, daily_returns |
| performances_metrics | dict[str, dict[str, float]] | Dictionary mapping strategy names to their corresponding performance metric dictionary                                  |

### Methods

| Method                                                                                      | Description                                                                  |
| ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| plot_comparison_equity_curve(title="Comparison Equity Curve", plot_overlapping_only=True)   | Generates an interactive equity curve line chart for each strategy.          |
| comparison_metrics_table(title="Comparison Performance Metrics")                            | Displays a table comparing performance metrics across strategies.            |
| plot_comparison_daily_returns(title="Comparison Daily Returns", plot_overlapping_only=True) | Generates a line chart showing daily returns for each strategy.              |
| plot(all=True, charts=None)                                                                 | Generates charts for either all or a specific list of supported chart types. |
| export(format="html", all=True, charts=None, export_dir="exports")                          | Exports charts to files in the specified format.                             |
