## Strategy

The `strategy` module defines the framework for trading strategies used in the backtesting system. It contains a `StrategyBase` class that all strategies should inherit from and implement their own trading signal logic.

### `StrategyBase()` Class

An abstract base class for all trading strategies. All custom strategies should inherit from this class and override the `generate_trading_signal` method.

#### Parameter

- `df_historical_data (pd.DataFrame, optional)`: Historical price data used for strategy computation.

#### Methods

`generate_trading_signal(data, datetime=None, index=0)`
To be implemented by subclasses. Returns one of the values in `Signal` enum:

- `Signal.BUY (1)`
- `Signal.SELL (-1)`
- `Signal.HOLD (0)`

### `BuyAndHoldStrategy()` Class

A simple strategy that issues a single buy signal on the first execution and holds the position forever.

### `MomentumStrategy()` Class

Calculates average momentum over a specified window.

- Buys if `momentum` > 0
- Sells if `momentum` < 0
- Holds if `momentum` = 0 or not available

#### Parameters

| Parameter          | Type      | Description                                |
| ------------------ | --------- | ------------------------------------------ |
| df_historical_data | dataframe | DataFrame with at least a 'close' column.  |
| window             | int       | Look-back period for momentum calculation. |

### `MovingAverageCrossoverStrategy()` Class

Implements a moving average crossover strategy.

- Buys when `SMA` > `LMA`
- Sells when `LMA` > `SMA`
- Holds when equal or not available

#### Parameters

| Parameter    | Type     | Description                           |
| ------------ | -------- | ------------------------------------- |
| short_window | Optional | Period for short-term moving average. |
| long_window  | Optional | Period for long-term moving average.  |

### `SampleStrategy()` Class

A toy/sample strategy integrating a custom FinalAlphaModel.
Currently, issues a buy signal if the closing price is below 60, else sells.

_Notes:_

1. All strategies must return a signal from the `Signal` enum: `BUY`, `SELL`, or `HOLD`.
2. `data` is assumed to be a dictionary or record of relevant market data for a given time step.
3. `datetime` is used to index into precomputed historical indicators.
