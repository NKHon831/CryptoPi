This module provides classes to manage trades, track performance, and evaluate strategies in a backtesting engine.

## Portfolio

Handles tracking and management of trading activities and account status during backtesting.

### Attributes
- `wallet`: Current cash in the portfolio.
- `holdings`: Asset units currently held.
- `equity`: Total value of wallet + holdings.
- `initial_capital`: Starting capital.
- `investment_rate`: Portion of capital allocated per trade.
- `shorting_preference`: Placeholder for future shorting strategy logic.
- `pending_orders`, `executed_orders`, `cancelled_orders`: Lists to track orders at various stages.
- `open_trades`, `closed_trades`: Lists to track trade lifecycle.
- `positions`: Stores position history.
- `performance`: Placeholder for performance metrics.
- `max_equity`: Peak equity for drawdown calculation.
- `total_signal`, `total_trading_signal`: For signal frequency analysis.

### Key Methods
- `get_equity_value(market_price=0.0)`: Calculates current equity based on market price.
- `has_open_trades(market_entry_type=None)`: Checks for active trades (optionally by type).
- `add_pending_order(order)`, `add_executed_order(order)`, `add_cancelled_order(order)`: Adds single orders.
- `add_open_trade(trade)`, `add_closed_trade(trade)`: Adds trades.
- `add_executed_orders(list[Order])`, `add_cancelled_orders(list[Order])`: Adds bulk orders.
- `get_pending_orders()`, `get_executed_orders()`, `get_open_trades()`, `get_closed_trades()`: Accessors for trade state.
- `get_all_trades()`: Returns all open and closed trades.
- `overview()`: Prints a summary of the portfolio status.

## PortfolioManager

Handles execution flow, trade tracking, position management, and performance analytics.

### Attributes
- `portfolio`: Instance of the `Portfolio` class.

### Key Methods
- `generate_order(symbol, trading_signal, current_market_data)`: Generates a pending order.
- `send_pending_orders()`: Returns and clears all pending orders.
- `update_orders(orders)`: Updates wallet and holdings based on executed or cancelled orders.
- `update_trades(executed_orders)`: Closes or creates new trades from order list.
- `update_portfolio(previous_data, current_data)`: Updates portfolio state (position, PnL, drawdown, equity).
- `export_closed_trades()`: Returns closed trades in a structured exportable format.
- `get_max_drawdown()`: Returns maximum drawdown observed during backtest.
- `calculate_sharpe_ratio(periods_per_year=252, risk_free_rate=0.0)`: Calculates Sharpe Ratio based on equity curve.


## Position Class

Represents the state of a single position in the portfolio.

### Attributes
- `direction`: Direction of trade (1 = long, -1 = short, 0 = neutral).
- `size`: Quantity of asset held.
- `pnl`: Profit or loss for the position.
- `equity`: Portfolio equity at that moment.
- `drawdown`: Equity drawdown from the historical peak.

### Methods
- `__str__()`: Returns string summary of the position’s details.

