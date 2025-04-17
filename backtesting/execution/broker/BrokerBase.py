from ..execution_success_model import ExecutionSuccessModel
from backtesting.execution.Trade import Trade
from backtesting.constants import OrderExecutionStatus
import random

class BrokerBase:
    def __init__(
            self,
            trading_fee : float = 0.0006, # percentage
            execution_success_model : ExecutionSuccessModel = ExecutionSuccessModel(),
    ):
        self.trading_fee = trading_fee
        self.execution_success_model = execution_success_model

    def execute_orders(self, orders):
        executed_orders = []
        cancelled_orders = []
        active_trades = []
        closed_trades = []

        for order in orders:

            # Need to define logic to determine whether open or close trade. For now all trades are open trades.
            result = self.execute_order(order)
            if (result['status'] == OrderExecutionStatus.OPEN):
                executed_orders.append(order)
                active_trades.append(result['trade'])

            elif (result['status'] == OrderExecutionStatus.CLOSED):
                executed_orders.append(order)
                closed_trades.append(result['trade'])

            elif (result['status'] == OrderExecutionStatus.FAILED):
                cancelled_orders.append(order)

        return {
            'executed_orders': executed_orders,
            'cancelled_orders': cancelled_orders,
            'active_trades': active_trades,
            'closed_trades': closed_trades,
        }

    def execute_order(self, order):
        result = {
            'status': OrderExecutionStatus.FAILED,
            'trade': None,
        }

        if(self.execution_success_model.is_order_executed_successfully()):
            # Have to implement logic to determine whether is open/close trade. Placeholder logic for now
            is_open_trade = True if random.random() > 0.5 else False
            if (is_open_trade):
                result['status'] = OrderExecutionStatus.OPEN
                trade = Trade.open_trade(order)
            else :
                result['status'] = OrderExecutionStatus.CLOSED

                trade = Trade.close_trade(order)
            result['trade'] = trade

        return result
    # Implement order execution logic here