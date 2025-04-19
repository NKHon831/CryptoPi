
class Position():
    def __init__(
            self,
            direction,
            size = 0,
            pnl = 0,
            equity = 0,
            drawdown = 0
        ):
        self.direction = direction
        self.size = size   
        self.pnl = pnl
        self.equity = equity
        self.drawdown = drawdown


    def __str__(self):
        return (
            f"Position:\n"
            f"Direction: {self.direction}\n"
            f"Size: {self.size}\n"
            f"PnL: {self.pnl}\n"
            f"Equity: {self.equity}\n"
            f"Drawdown: {self.drawdown}\n"
        )