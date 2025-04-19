## PerformanceManager

The `PerformanceManager` is the central component responsible for computing various performance metrics for a backtested trading strategy. It aggregates results from multiple submodules to provide both scalar and time-series metrics in a clean and structured format.

### Parameters

| Parameter       | Type       | Purpose                                                                           |
| --------------- | ---------- | --------------------------------------------------------------------------------- |
| trades          | List(Dict) | A list of executed trade records, each being a dictionary with trade information. |
| initial_capital | float      | The starting capital of the portfolio                                             |
| benchmark_curve | optional   | Time-series data used for benchmarking performance                                |

### Methods

`get_metrics()` returns:

- `scalar_metrics` (pd.DataFrame): A one-row dataframe containing scalar metrics, such as total return, Sharpe ratio, etc.
- `time_series_metrics` (Dict): A dictionary of time series values for visualization (e.g., equity curve, drawdowns).

## PerformanceBase

The `PerformanceBase` class provides foundational calculations and data enrichment for performance analysis. It is initialized with a list of trade dictionaries and an initial capital value. The class appends transaction fees and computes the PnL (Profit and Loss) for each trade, forming the base data used by all metric calculators.

### Parameters

| Parameter       | Type       | Purpose                                                                           |
| --------------- | ---------- | --------------------------------------------------------------------------------- |
| trades          | List(Dict) | A list of executed trade records, each being a dictionary with trade information. |
| initial_capital | float      | The starting capital of the portfolio                                             |
| fee_rate        | Optional   | Transaction fee rate in percentage. Defaults to 0.06.                             |

### Methods

| Method                        | Purpose                                                                                       |
| ----------------------------- | --------------------------------------------------------------------------------------------- |
| append_fee(rate, type="both") | Adds a "fee" field to each trade based on fee type                                            |
| append_pnl()                  | Calculates the net PnL per trade, adjusted for direction and fees, and appends a "pnl" field. |

## Metric Types

### 1. TradeMetrics

The TradeMetrics submodule is responsible for analyzing trade-level statistics, providing insights into the quality and characteristics of individual trades.

#### Metrics

| Metric                | Description                                           |
| --------------------- | ----------------------------------------------------- |
| total_trades          | Total number of trades executed.                      |
| win_rate              | Proportion of profitable trades.                      |
| loss_rate             | Proportion of unprofitable (losing) trades.           |
| average_win           | Mean profit for winning trades.                       |
| average_loss          | Mean loss for losing trades.                          |
| holding_period_mean   | Average duration (in time units) a position was held. |
| holding_period_median | Median duration a position was held.                  |
| holding_period_max    | Longest duration a single trade was held.             |
| holding_period_min    | Shortest duration a single trade was held.            |
| profit_factor         | Ratio of gross profit to gross loss.                  |

#### Methods

`calculate_all()` calculates and returns all above metrics.

### 2. TimeSeriesMetrics

#### Metrics

| Metric | Description |
| ------ | ----------- |

#### Methods

`calculate_all()` calculates and returns all above metrics.

### 3. RiskMetrics

#### Metrics

| Metric                   | Description                                                                 |
| ------------------------ | --------------------------------------------------------------------------- |
| sharpe_ratio             | Sharpe ratio using daily returns: excess return over standard deviation.    |
| annualized_sharpe_ratio  | Sharpe ratio annualized over 252 trading days.                              |
| sortino_ratio            | Sortino ratio using downside deviation instead of total standard deviation. |
| annualized_sortino_ratio | Annualized Sortino ratio over 252 trading days.                             |
| calmar_ratio             | Calmar ratio: annualized return divided by maximum drawdown.                |

#### Methods

`calculate_all()` calculates and returns all above metrics.

### 4. ComparativeMetrics

#### Metrics

| Metric         | Description                                                                             |
| -------------- | --------------------------------------------------------------------------------------- |
| alpha          | The strategy's excess return over the benchmark, annualized.                            |
| beta           | Sensitivity of the strategy’s returns to the benchmark’s returns.                       |
| tracking_error | Annualized standard deviation of the difference between strategy and benchmark returns. |

#### Methods

`calculate_all()` calculates and returns all above metrics.
