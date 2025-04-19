## Visualisation

The `Visualisation` class is responsible for generating and displaying various performance-related charts and metrics for a backtested trading strategy. It integrates with the PerformanceManager to visualize the strategy's performance, including the equity curve, daily returns, performance metrics, and price chart.

To create an instance of the class:

```python
from backtesting.visualisation.visualisation import StrategyVisualisation
strategyVisualiser = StrategyVisualisation(performance_data, performance_metrics, market_data)
```

### Parameters

| Parameter           | Type                       | Description                                                                                                                                        |
| ------------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| performance_data    | dict[str, dict[str, list]] | A dictionary where each key is a strategy name and the value is a dictionary with keys like 'datetimes', 'equity_values', and 'daily_returns'.     |
| performance_metrics | pd.DataFrame or pd.Series  | A set of scalar performance metrics (e.g., Sharpe ratio, max drawdown) to be displayed in a summary table.                                         |
| market_data         | pd.DataFrame               | A DataFrame containing market data with columns such as 'datetime', 'open', 'high', 'low', and 'close'. This data will be used to plot the charts. |

### Methods

| Method                                                                   | Description                                                                |
| ------------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| plot_equity_curve(self, title="Equity Curve")                            | Plots the equity curve for each strategy in the performance_data.          |
| tabulate_metrics(self, title="Performance Metrics Summary")              | Creates a table displaying the performance metrics for the strategy.       |
| plot_daily_returns(self, title="Daily Returns")                          | Plots the daily returns for each strategy in the performance_data.         |
| plot_price(self, title="Price Chart", with_signals=True)                 | Plots the price chart along with buy and sell signals.                     |
| plot(self, all=True, charts=None)                                        | Plots the requested charts either all at once or a specific set of charts. |
| export(self, format="html", all=True, charts=None, export_dir="exports") | Exports the charts in the specified format to the specified directory.     |

## ComparisonPlot

The `ComparisonPlot` class provides a suite of interactive charts using Plotly for comparing the performance of multiple backtesting strategies.

### Parameters

| Parameter            | Type                        | Description                                                                                                              |
| -------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| performances_data    | dict[str, pd.DataFrame]     | Dictionary where keys are strategy names and values are DataFrames with columns datetimes, equity_values, daily_returns. |
| performances_metrics | dict[str, dict[str, float]] | Dictionary mapping strategy names to their corresponding performance metric dictionary.                                  |

### Methods

| Method                                                                                      | Description                                                                  |
| ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| plot_comparison_equity_curve(title="Comparison Equity Curve", plot_overlapping_only=True)   | Generates an interactive equity curve line chart for each strategy.          |
| comparison_metrics_table(title="Comparison Performance Metrics")                            | Displays a table comparing performance metrics across strategies.            |
| plot_comparison_daily_returns(title="Comparison Daily Returns", plot_overlapping_only=True) | Generates a line chart showing daily returns for each strategy.              |
| plot(all=True, charts=None)                                                                 | Generates charts for either all or a specific list of supported chart types. |
| export(format="html", all=True, charts=None, export_dir="exports")                          | Exports charts to files in the specified format.                             |
