This module handles how orders are executed, tracked, and converted into trades within a backtesting environment.

## BrokerBase

Simulates a broker executing orders in a market environment using a success model and trading fees.

### Attributes
- `trading_fee (float)`: Percentage-based trading fee (default: 0.0006 or 0.06%).
- `execution_success_model (ExecutionSuccessModel)`: Determines whether an order is successfully executed.

### Methods

#### `execute_orders(orders: list[Order], wallet, market_data, datetime)`: Executes a list of orders using current market data and broker logic.

- **Returns**: 
  ```python
  {
    'executed_orders': [...],
    'cancelled_orders': [...]
  }

`execute_order(order: Order, wallet, market_data, datetime)`: Executes a single order based on the execution_success_model.

- **Returns**: `Order` object with updated status, executed_price, and executed_date_time.

## Order
Represents a trade instruction with parameters like quantity, price, and signal direction.

### Attributes
- `symbol (str)`: Asset symbol (e.g., BTCUSD).
- `trading_signal (Signal)`: BUY or SELL signal.
- `quantity (float`: Number of units to buy/sell.
- `price_limit (float, optional)`: Optional price limit for limit orders.
- `stop_loss_price (float, optional)`: Price to trigger a stop loss.
- `desired_price (float, optional)`: Ideal execution price.
- `execution_type (str)`: OPEN or CLOSE (default: OPEN).
- `execution_interval (str)`: Timeframe granularity (e.g., HOUR).
- `status (OrderStatus)`: PENDING, EXECUTED, or CANCELLED.
- `executed_date_time (datetime)`: Timestamp of execution.
- `executed_price (float)`: Price at which the order was filled.

### Methods
- `__str__()`: Returns a basic string summary of the order.

## Trade
Represents an actual position taken based on an executed order.

### Attributes
- `symbol (str)`: Asset involved in the trade.
- `entry_price (float)`: Price at which the trade was opened.
- `entry_time (datetime)`: Timestamp of entry.
- `quantity (float)`: Number of units traded.
- `market_entry_type (str)`: LONG or SHORT.
- `stop_loss_price (float)`: Stop loss trigger.
- `exit_price (float, optional)`: Price at which trade closed.
- `exit_time (datetime, optional)`: Timestamp of exit.
- `profit (float)`: Target profit (not realized).
- `status (TradeStatus)`: OPEN or CLOSED.

### Methods
- `__str__()`: Human-readable summary of the trade.
- `@staticmethod create(order: Order, quantity: float = None) -> Trade`: Creates a Trade instance from an executed Order.

## ExecutionSuccessModel
Controls the logic for whether an order will successfully execute.

### Attributes
- `sucess_rate (float)`: Probability of success (default is 1.0 = 100%).

### Methods
- `is_order_executed_successfully() -> bool`: Returns whether the current order execution attempt is successful.

## Execution Flow Summary
1. Strategy generates pending Orders.
2. BrokerBase.execute_orders receives these orders, applies fees, and determines execution success.
3. Orders that succeed become Trades.
4. PortfolioManager processes trades, updates equity, and tracks performance.