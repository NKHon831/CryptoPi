from ..execution_success_model import ExecutionSuccessModel
from backtesting.constants import OrderStatus, MarketEntryType
from backtesting.execution.Order import Order

class BrokerBase:
    def __init__(
            self,
            trading_fee : float = 0.0006, # percentage
            execution_success_model : ExecutionSuccessModel = ExecutionSuccessModel(),
    ):
        self.trading_fee = trading_fee
        self.execution_success_model = execution_success_model

    def execute_orders(self, orders : list[Order], wallet, market_data, datetime):
        executed_orders = []
        cancelled_orders = []
        for order in orders:
            resultOrder = self.execute_order(order, wallet, market_data, datetime)
            if (resultOrder.status == OrderStatus.EXECUTED):
                executed_orders.append(resultOrder)
            elif (resultOrder.status == OrderStatus.CANCELLED): 
                cancelled_orders.append(resultOrder)

        return {
            'executed_orders': executed_orders,
            'cancelled_orders': cancelled_orders,
        }

    def execute_order(self, order : Order, wallet, market_data, datetime):
        if(self.execution_success_model.is_order_executed_successfully()):
            order.status = OrderStatus.EXECUTED
            order.executed_price = market_data['open']
            order.executed_date_time = datetime 
            
        else: 
            order.status = OrderStatus.CANCELLED
        
        return order