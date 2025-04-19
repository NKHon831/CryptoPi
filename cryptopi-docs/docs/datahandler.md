This module provides a structured, extensible way to retrieve, process, and cache financial and crypto market data from various sources. It consists of a base class `BaseDataHandler` and two specialized subclasses: `RegimeModelData` and `FinalAlphaModelData`.

## BaseDataHandler
Handles the core logic for fetching raw data from various APIs, applying preprocessing functions, and caching the results using `joblib` and `requests_cache`.  

### Parameters

| Parameter       | Type                     | Description                                                                       |
| --------------- | ------------------------ | --------------------------------------------------------------------------------- |
| symbol          | string                   | Trading pair or asset (e.g., BTC)                                                 |
| start_time      | datetime                 | Start of data range (e.g., datetime(2024, 4, 17)).                                |
| end_time        | datetime                 | End of data range (e.g., datetime(2024, 4, 18)).                                  |
| window          | string                   | Timeframe for data (e.g., 1d, 1h, 15m).                                           |
| limit           | (int, default=100000)    | Maximum number of rows to fetch.                                                  |
| flatten         | (bool, default=True)     | If True, flattens nested JSON data when applicable.                               |

### Public Methods

- `fetch_binance_data()`: Downloads OHLCV data from Binance via Cybotrade Datasource API. Automatically uses `requests_cache` to cache results.
- `get_processed_data()`: Returns the processed pandas DataFrame.
- `export(path: str, filename: str)`: Saves the processed data to a CSV file at the given path.
- `load_from_disc(path: str)` : Loads raw data from a CSV file into the internal raw_data container.

### Internal Methods

- `convert_to_unix_ms(dt: datetime)` : Converts a datetime object to a UNIX timestamp in milliseconds.
- `_generate_cache_key()`: Returns a tuple used to identify cached results.
- `_get_cache_key()`: Returns a string-formatted version of the cache key.

### Example Usage
```python
from backtesting.datahandler import BaseDataHandler

baseDataHandler = BaseDataHandler(symbol='BTC-USD',
                      start_time=datetime(2025, 3, 1, tzinfo=timezone.utc),
                      end_time=datetime(2025, 4, 1, tzinfo=timezone.utc),
                      window="1h")
```

## RegimeModelData

Custom data handler for regime classification models. Applies a series of volatility, trend, momentum, and volume-based indicators to compute regime shifts in market behavior.

### Indicators

| Indicator             | Internal Method                   | Description                                                                       |
| --------------------  | ----------------------------------| --------------------------------------------------------------------------------- |
| Log Return            | (inline in `preprocess`)          | Measures relative price change; used to standardize returns.                      |
| Volatility (10)       | (inline in `preprocess`)          | Rolling standard deviation of log returns; used to assess price turbulence.       |
| Vol-Adj Return        | (inline in `preprocess`)          | Normalizes return by volatility; highlights abnormal movements.                   |
| EMA (50, 200)         | compute_ema, compute_ema_200      | Tracks price trend smoothness over short/long periods.                            |
| RSI (14)              | compute_rsi                       | Identifies overbought/oversold conditions.                                        |
| MACD + Signal         | compute_macd                      | Detects trend reversals via moving average crossovers.                            |
| ATR (14)              | compute_atr                       | Measures price range volatility for risk analysis.                                |
| SAR                   | compute_sar                       | Trend-following indicator showing potential reversals.                            |
| Slope (14)            | compute_slope                     | Measures directional rate of change.                                              |
| ADX (14)              | compute_adx                       | Quantifies trend strength regardless of direction.                                |
| OBV                   | compute_obv_vectorized            | Measures volume flow to validate price trends.                                    |

### Caching

- Loads from or saves to disk using `joblib` under `cache/processed/`.
- Automatically cleans up old cache files (older than 1 hour).

### Public Methods
Inherits all public methods from `BaseDataHandler`.

### Example Usage

```python
from backtesting.datahandler import RegimeModelData

regimeModelData = RegimeModelData(symbol='BTC-USD',
                      start_time=datetime(2025, 3, 1, tzinfo=timezone.utc),
                      end_time=datetime(2025, 4, 1, tzinfo=timezone.utc),
                      window="1h")
```

## LogisticRegressionModelData

Specialized class for fetching multiple alpha-relevant features from Glassnode using the Cybotrade Datasource API. Data is compiled into a single DataFrame.

### Indicators

#### 1. Addresses

| Indicator             | Description                                                                                                    |
| --------------------- | ---------------------------------------------------------------------------------------------------------------|
| min_10k_count         | Number of addresses holding at least 10,000 units of an asset. Indicates whale activity and market influence.  |
| min_100_count         | Number of addresses holding at least 100 units. Tracks retail accumulation.                                    |
| new_non_zero_count    | Number of new addresses with non-zero balance. Proxy for network growth and adoption.                          |
| accumulation_count    | Addresses consistently accumulating more coins over time. Signals strong investor conviction.                  |
| count                 | Total number of addresses. General measure of network size.                                                    |
| sopr                  | Spent Output Profit Ratio. Ratio of price sold vs price paid for coins moved. Values >1 indicate profit taking.|

#### 2. Blockchain

| Indicator             | Description                                                                                       |
| --------------------- | ------------------------------------------------------------------------------------------------- |
| block_count           | Number of blocks mined. Indicates network liveness.                                               |
| utxo_created_count    | Number of unspent transaction outputs created. Reflects transaction activity and coin movement.   |

#### 3. Distribution

| Indicator             | Description                                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------------------- |
| balance_exchanges     | Aggregate balance held on exchanges. Tracks potential sell-side liquidity and risk-on/risk-off sentiment. |

#### 4. Mining

| Indicator             | Description                                                                                   |
| --------------------- | --------------------------------------------------------------------------------------------- |
| hash_rate_mean        | Average hash rate over time. Proxy for network security and miner sentiment.                  |
| revenue_from_fees     | Miner revenue from transaction fees. Reflects demand for block space.                         |
| realized_loss         | Total value of coins moved at a lower price than they were acquired. Tracks panic selling.    |
| realized_profit       | Total value of coins moved at a higher price than acquisition. Indicates profit realization.  |

#### 5. Indicators

| Indicator                     | Description                                                                                   |
| ------------------------------| --------------------------------------------------------------------------------------------- |
| net_realized_profit_loss      | Difference between realized profit and loss. Signals net capital flows on-chain.              |
| net_unrealized_profit_loss    | Unrealized gains/losses across all held coins. Used to identify market euphoria or fear.      |
| liveliness                    | Ratio of total coin days destroyed to total coin days created. Indicates HODLing behavior.    |
| nvt                           | Network Value to Transactions ratio. Higher values suggest overvaluation relative to usage.   |
| nvts                          | NVT Signal: smoothed version of NVT for better signal extraction.                             |
| reserve_risk                  | Risk/reward ratio based on HODL waves and price. Useful for long-term valuation.              |
| rhodl_ratio                   | Combines age bands of coins to detect overheated or undervalued markets.                      |
| seller_exhaustion_constant    | Measures when price and miner sell pressure are both low. Bullish reversal signal.            |
| stock_to_flow_deflection      | Deviation from S2F model. Gauges scarcity relative to issuance.                               |
| velocity                      | Rate of turnover of coins in the network. High values indicate economic activity.             |

#### 6. Supply

| Indicator                 | Description                                                                           |
| ------------------------- | ------------------------------------------------------------------------------------- |
| active_more_1y_percent    | Percentage of coins unmoved for over 1 year. Indicates long-term holder conviction.   |
| inflation_rate            | Annualized increase in circulating supply. Important for monetary policy modeling.    |
| loss_sum                  | Aggregate USD value of all coins currently in loss. Shows unrealized downside risk.   |

#### 7. Transactions

| Indicator                                     | Description                                                                                       |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| transfers_to_exchanges_count_pit              | Count of transactions sent to exchanges. Can indicate rising sell pressure.                       |
| transfers_volume_to_exchanges_sum             | Total volume sent to exchanges. Tracks potential supply-side liquidity.                           |
| transfers_volume_exchanges_net_pit            | Net transfer volume in/out of exchanges. Positive values indicate accumulation.                   |
| transfers_volume_from_exchanges_mean_pit      | Average volume withdrawn from exchanges. High values indicate investor confidence.                |
| transfers_volume_from_exchanges_sum_pit       | Total volume withdrawn from exchanges. Measures HODLer behavior.                                  |
| transfers_from_exchanges_count_pit            | Number of withdrawal transactions. Correlates with exchange outflows.                             |
| transfers_volume_entity_adjusted_sum_pit      | Total adjusted transfer volume across all entities. Filters internal transfers.                   |
| transfers_volume_within_exchanges_sum_pit     | Volume moving between wallets within the same exchange. Tracks internal flow.                     |
| transfers_volume_between_exchanges_sum_pit    | Volume moved between different exchanges. Reflects inter-exchange arbitrage and liquidity shift.  |
 
#### Public Methods

- `fetch_all_endpoints()`: Aggregates data from all endpoints using common parameters and returns a single DataFrame.

### Example Usage

```python
from backtesting.datahandler import LogisticRegressionModelData

logisticRegressionModel = LogisticRegressionModelData(symbol='BTC-USD',
                      start_time=datetime(2025, 3, 1, tzinfo=timezone.utc),
                      end_time=datetime(2025, 4, 1, tzinfo=timezone.utc),
                      window="1h")
```

## BenchmarkData

A lightweight benchmark data handler designed to fetch historical closing price data from Yahoo Finance using the yfinance library. Primarily used to serve as a market benchmark (e.g., BTC/USD) for performance comparison in model backtesting.

| Parameter       | Type                     | Description                                                                       |
| --------------- | ------------------------ | --------------------------------------------------------------------------------- |
| symbol          | string                   | Trading pair or asset (e.g., BTC-USD)                                                 |
| start_time      | datetime                 | Start of data range (e.g., datetime(2024, 4, 17)).                                |
| end_time        | datetime                 | End of data range (e.g., datetime(2024, 4, 18)).                                  |
| interval        | string                   | Timeframe for data (e.g., 1m, 60m, 1d)                                            |

#### Public Methods
`fetch_yfinance_data()` : Fetches benchmark price data for the given symbol and date range using yfinance.

### Catching Strategy

#### `requests_cache`
- Caches raw API responses on disk to minimize duplicate requests.
- Useful for paginated and repetitive queries.
- Used in fetching OHLC

#### `joblib`

- Caches processed DataFrames based on parameters (symbol, endpoint, start/end date, interval).
- Reduces recomputation for common queries.