from ..execution.Order import Order
from ..execution.Trade import Trade
from ..constants import MarketEntryType

class Portfolio:
    def __init__(
            self, 
            holdings = 0.0, 
            wallet = 10000,
            investment_rate = 0.01, # 1% of the portfolio equity/wallet is used for each order
            shorting_preference = 1 # All holdings is used for shorting / Assume short is just sell what we hold
        ):
        self.pending_orders  = []
        self.cancelled_orders = []
        self.executed_orders = []
        self.open_trades = {
            MarketEntryType.LONG : [],
            MarketEntryType.SHORT : []
        }
        self.closed_trades = []
        self.performance = None # Implement performance evaluation
        self.position = None # Dont support position yet
        self.wallet = wallet
        self.holdings = holdings
        self.equity = self.get_equity_value()
        self.investment_rate = investment_rate
        self.shorting_preference = shorting_preference
    
    def get_equity_value(self, market_price = 0.0):
        return self.wallet + self.holdings * market_price

    def has_open_trades(self):
        long_open_trades = self.open_trades[MarketEntryType.LONG]
        short_open_trades = self.open_trades[MarketEntryType.SHORT]
        
        return long_open_trades or short_open_trades

    # Add one order/trade
    def add_pending_order(self, pending_order : Order):
        self.pending_orders.append(pending_order)
    
    def add_executed_order(self, executed_order : Order):
        self.executed_orders.append(executed_order)
    
    def add_cancelled_order(self, cancelled_order : Order):
        self.cancelled_orders.append(cancelled_order)

    def add_open_trade(self, trade : Trade):
        self.open_trades[trade.market_entry_type].append(trade)
    
    def add_closed_trade(self, closed_trade : Trade):
        self.closed_trades.append(closed_trade)

    # Add multiple orders/trades
    def add_executed_orders(self, executed_orders : list[Order]):
        self.executed_orders.extend(executed_orders)
    
    def add_cancelled_orders(self, cancelled_orders : list[Order]):
        self.cancelled_orders.extend(cancelled_orders)
    
    def get_pending_orders(self):
        return self.pending_orders

    def get_cancelled_orders(self):
        return self.cancelled_orders
    
    def get_executed_orders(self):
        return self.executed_orders
    
    def get_open_trades(self):
        return self.open_trades
    
    def get_closed_trades(self):
        return self.closed_trades
    
    def get_all_trades(self):
        return (self.open_trades[MarketEntryType.LONG] + self.open_trades[MarketEntryType.SHORT] + self.closed_trades)

    def overview(self):
        print('Pending Orders:', len(self.pending_orders))
        print('Cancelled Orders:', len(self.cancelled_orders))
        print('Executed Orders:', len(self.executed_orders))
        print('Open trades:')
        print("LONG: ", len(self.open_trades[MarketEntryType.LONG]))
        print("SHORT: ", len(self.open_trades[MarketEntryType.SHORT]))
        print('Closed trades:', len(self.closed_trades))
