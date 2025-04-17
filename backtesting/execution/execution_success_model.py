class ExecutionSuccessModel:
    def __init__(
            self,
            sucess_rate: float = 1.0,  # 100% success rate by default
    ):
        pass

    def is_order_executed_successfully(self) -> bool :
        # Implement the exact success model logic here
        return True