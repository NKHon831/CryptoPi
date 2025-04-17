from ..constants import ExecutionType , ExecutionInterval

class Order:
    def __init__(
            self,
            execution_type : str = ExecutionType.OPEN,
            execution_interval : str = ExecutionInterval.HOUR,
    ):
        pass

    # static method to create an order object
    def create(data):   
        return Order()