from ..constants import ExecutionType , ExecutionInterval
from ..constants import OrderStatus

class Order:
    def __init__(
            self,
            quantity : float, # Amount of units to buy/sell
            price_limit: float, # Price limit for the order
            market_entry_type : str, # buy/sell
            stop_loss_price : float, # Stop loss price
            desired_price : float, # Desired price for the order
            execution_type : str = ExecutionType.OPEN,
            execution_interval : str = ExecutionInterval.HOUR,
            status : OrderStatus = OrderStatus.PENDING,
    ):
        self.quantity = quantity
        self.price_limit = price_limit
        self.market_entry_type = market_entry_type
        self.stop_loss_price = stop_loss_price
        self.desired_price = desired_price
        self.execution_type = execution_type
        self.execution_interval = execution_interval
        self.status = status

    # static method to create an order object
    def create(data):   
        return Order(1, 1, 'LONG', 1, 1)