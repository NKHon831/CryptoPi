from .execution.Order import Order
from .execution.Trade import Trade

class Portfolio:
    def __init__(self, holdings = 0.0, wallet = 0.0):
        self.pending_orders  = []
        self.cancelled_orders = []
        self.executed_orders = []
        self.active_trades = []
        self.closed_trades = []
        self.performance = None # Implement performance evaluation
        self.position = None # Dont support position yet
        self.wallet = wallet
        self.holdings = holdings
    
    def get_performance(self):
        # Retrieve portfolio's performance 
        pass

    def add_pending_order(self, pending_order : Order):
        self.pending_orders.append(pending_order)
    
    def add_executed_order(self, executed_order : Order):
        self.executed_orders.append(executed_order)
    
    def add_cancelled_order(self, cancelled_order : Order):
        self.cancelled_orders.append(cancelled_order)

    def add_active_trade(self, active_trade : Trade):
        self.active_trades.append(active_trade)
    
    def add_closed_trade(self, closed_trade : Trade):
        self.closed_trades.append(closed_trade)
    
    def get_pending_orders(self):
        return self.pending_orders
    
    def send_pending_orders(self):
        pending_orders = self.pending_orders.copy()
        self.pending_orders.clear()
        return pending_orders

    def get_cancelled_orders(self):
        return self.cancelled_orders
    
    def get_executed_orders(self):
        return self.executed_orders
    
    def get_active_trades(self):
        return self.active_trades
    
    def get_closed_trades(self):
        return self.closed_trades

    def buy(self):
        #implement buy logic here
        return 1
    
    def hold(self):
        #implement hold logic here
        return 0
    
    def sell(self):
        #implement sell logic here
        return -1
    
    def update_records(self, records):
        self.executed_orders.extend(records['executed_orders'])
        self.cancelled_orders.extend(records['cancelled_orders'])
        self.active_trades.extend(records['active_trades'])    
        self.closed_trades.extend(records['closed_trades'])

    def overview(self):
        print('Pending Orders:', len(self.pending_orders))
        print('Cancelled Orders:', len(self.cancelled_orders))
        print('Executed Orders:', len(self.executed_orders))
        print('Active trades:', len(self.active_trades))
        print('Closed trades:', len(self.closed_trades))
