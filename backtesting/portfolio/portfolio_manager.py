from backtesting.portfolio.portfolio import Portfolio
from backtesting.execution.Order import Order
from backtesting.execution.Trade import Trade
from backtesting.constants import MarketEntryType, TradeStatus

class PortfolioManager:

    def __init__(
            self, 
            portfolio : Portfolio = Portfolio()
        ):
        self.portfolio = portfolio

    def send_pending_orders(self):
        pending_orders = self.portfolio.pending_orders.copy()
        self.portfolio.pending_orders.clear()
        return pending_orders

    def update_orders(self, orders):
        executed_orders = orders['executed_orders']
        cancelled_orders = orders['cancelled_orders']
        
        self.portfolio.add_executed_orders(executed_orders)
        self.portfolio.add_cancelled_orders(cancelled_orders)

        self.update_trades(executed_orders)

    def update_trades(self, executed_orders : list[Order]):
        for order in executed_orders:
            if self.portfolio.has_open_trades():

                # See if can close any open trades
                match order.market_entry_type : 
                    case MarketEntryType.LONG : 
                        short_open_trades = self.portfolio.get_open_trades()[MarketEntryType.SHORT]
                        self.closing_trade(order, short_open_trades)

                    case MarketEntryType.SHORT:
                        long_open_trades = self.portfolio.get_open_trades()[MarketEntryType.LONG]
                        self.closing_trade(order, long_open_trades)

                    # Dont have a matching trade to close
                    case _:
                        new_trade = Trade.create(order)
                        self.portfolio.add_open_trade(new_trade)

            elif not self.portfolio.has_open_trades():
                new_trade = Trade.create(order)
                self.portfolio.add_open_trade(new_trade)
    
    def closing_trade(self, executed_order : Order, open_trades : list[Trade]):
        order_quantity_to_close = executed_order.quantity

        while order_quantity_to_close > 0:
            if(len(open_trades) == 0):
                new_trade = Trade.create(executed_order, order_quantity_to_close)
                self.portfolio.add_open_trade(new_trade)
                break

            trade_to_close = open_trades[0]

            traded_quantity = min(trade_to_close.quantity, order_quantity_to_close)
            order_quantity_to_close -= traded_quantity

            # Update the trade and close if trade.quantity is 0
            trade_to_close.quantity = trade_to_close.quantity - traded_quantity

            if(trade_to_close.quantity == 0 ):
                trade_to_close.profit = traded_quantity * (executed_order.executed_price - trade_to_close.entry_price)
                trade_to_close.status = TradeStatus.CLOSED
                trade_to_close.exit_time.append(executed_order.executed_date_time)
                trade_to_close.exit_price.append(executed_order.executed_price)

                # Close the trade and move it from open_trades to closed_trades
                closed_trade = open_trades.pop()
                self.portfolio.add_closed_trade(closed_trade)


                

            
            
                








   
