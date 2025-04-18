from backtesting.portfolio.portfolio import Portfolio
from backtesting.execution.Order import Order

class PortfolioManager:

    def __init__(self):
        pass

    @staticmethod
    def update_orders(portfolio : Portfolio, orders):
        executed_orders = orders['executed_orders']
        cancelled_orders = orders['cancelled_orders']
        
        portfolio.add_executed_orders(executed_orders)
        portfolio.add_cancelled_orders(cancelled_orders)

        PortfolioManager.update_trades(portfolio, executed_orders)

    @staticmethod
    def update_trades(portfolio : Portfolio, executed_orders):

        
        pass
