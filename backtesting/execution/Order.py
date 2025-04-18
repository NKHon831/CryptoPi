from ..constants import ExecutionType , ExecutionInterval, OrderStatus, MarketEntryType, Signal
from datetime import datetime

class Order:
    def __init__(
            self,
            quantity : float, # Amount of units to buy/sell
            price_limit: float, # Price limit for the order
            trading_signal : str, # buy/sell
            stop_loss_price : float, # Stop loss price
            desired_price : float, # Desired price for the order
            execution_type : str = ExecutionType.OPEN,
            execution_interval : str = ExecutionInterval.HOUR,
            status : OrderStatus = OrderStatus.PENDING,
            executed_date_time : datetime = None,
            executed_price : float = None,
    ):
        self.quantity = quantity
        self.price_limit = price_limit
        self.market_entry_type = MarketEntryType.LONG if trading_signal is Signal.BUY else MarketEntryType.SHORT
        self.stop_loss_price = stop_loss_price
        self.desired_price = desired_price
        self.execution_type = execution_type
        self.execution_interval = execution_interval
        self.status = status
        self.executed_date_time = executed_date_time
        self.executed_price = executed_price

    def __str__(self):
        return f"Order: \nExecuted price:   {self.executed_price}"
