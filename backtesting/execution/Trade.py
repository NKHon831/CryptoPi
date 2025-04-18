from backtesting.execution.Order import Order
from backtesting.constants import TradeStatus

class Trade:
    def __init__(
            self,
            entry_price : float, # Price at which the trade was opened
            exit_price : float, # Price at which the trade was closed
            entry_time : str, # Time at which the trade was opened
            exit_time : str, # Time at which the trade was closed
            quantity : float, # Amount of units traded
            market_entry_type : str, # Direction of the trade (buy/sell)
            stop_loss_price : float, # Stop loss price
            profit : float, # Profit target price
            status : TradeStatus = TradeStatus.OPEN, # Status of the trade (open/closed)
    ):
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.entry_time = entry_time
        self.exit_time = exit_time
        self.quantity = quantity
        self.market_entry_type = market_entry_type
        self.stop_loss_price = stop_loss_price
        self.profit = profit
        self.status = status

