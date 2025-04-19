from ..BrokerBase import BrokerBase
from ...execution_success_model import ExecutionSuccessModel

class DefaultBroker(BrokerBase):
    def __init__(
            self,
            trading_fee : float = 0.0006, # percentage
            execution_success_model : ExecutionSuccessModel = ExecutionSuccessModel(),
    ):
        super().__init__(trading_fee, execution_success_model)


    # Use the BrokerBase execute_order method to execute the order
    # def execute_order(self, order):
    #     pass