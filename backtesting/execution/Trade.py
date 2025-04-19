from backtesting.execution.Order import Order
from backtesting.constants import TradeStatus
from datetime import datetime
from backtesting.constants import MarketEntryType, Signal

class Trade:
    def __init__(
            self,
            symbol : str,
            entry_price : float, # Price at which the trade was opened
            entry_time : datetime, # Time at which the trade was opened
            quantity : float, # Amount of units traded
            market_entry_type : str, # Direction of the trade (buy/sell)
            stop_loss_price : float, # Stop loss price
            exit_price : float = None, # Price at which the trade was closed
            exit_time : datetime = None, # Time at which the trade was closed
            profit : float = 0.0, # Profit target price
            status : TradeStatus = TradeStatus.OPEN, # Status of the trade (open/closed)
    ):
        self.symbol = symbol
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.entry_time = entry_time
        self.exit_time = exit_time
        self.quantity = quantity
        self.market_entry_type = market_entry_type
        self.stop_loss_price = stop_loss_price
        self.profit = profit
        self.status = status
        self.symbol = symbol
    
    def __str__(self):
        return (
            f"Trade:\n"
            f"Entry price: {self.entry_price}\n"
            f"Entry time: {self.entry_time}\n"
            f"Quantity: {self.quantity}\n"
            f"Market entry type: {self.market_entry_type}\n"
            f"Stop loss price: {self.stop_loss_price}\n"
            f"Exit price: {self.exit_price}\n"
            f"Exit time: {self.exit_time}\n"
            f"Profit: {self.profit}\n"
            f"Status: {self.status}\n"
        )

    @staticmethod
    def create(order : Order, quantity : float = None):
        return Trade(
                    order.symbol,
                    order.executed_price,
                    order.executed_date_time,
                    quantity if quantity is not None else order.quantity,
                    MarketEntryType.LONG if order.trading_signal is Signal.BUY else MarketEntryType.SHORT,
                    order.stop_loss_price,
                )

