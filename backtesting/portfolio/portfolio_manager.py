from backtesting.portfolio.portfolio import Portfolio
from backtesting.execution.Order import Order
from backtesting.execution.Trade import Trade
from backtesting.constants import MarketEntryType, TradeStatus, Signal
import copy

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
    
    def generate_order(self, trading_signal : Signal, current_market_data):
        quantity_to_trade = self.calculate_quantity_to_trade(trading_signal, current_market_data)

        # No price_limit, stop_loss price, desired_price logic
        new_order = Order(trading_signal, quantity_to_trade)
        self.portfolio.add_pending_order(new_order)

    def update_orders(self, orders):
        executed_orders : list[Order] = orders['executed_orders']
        cancelled_orders : list[Order] = orders['cancelled_orders']

        for order in executed_orders: 
            match order.market_entry_type:
                case MarketEntryType.LONG:
                    self.portfolio.wallet -= order.quantity * order.executed_price
                    self.portfolio.holdings += order.quantity
                case MarketEntryType.SHORT:
                    self.portfolio.wallet += order.quantity * order.executed_price
                    self.portfolio.holdings -= order.quantity
                case _:
                    pass

        self.portfolio.add_executed_orders(executed_orders)
        self.portfolio.add_cancelled_orders(cancelled_orders)

        self.update_trades(executed_orders)

    def update_trades(self, executed_orders : list[Order]):

        for order in executed_orders:
            # Order is closing an open short trade
            if(order.market_entry_type is MarketEntryType.LONG and self.portfolio.has_open_trades(MarketEntryType.SHORT)):
                short_open_trades = self.portfolio.get_open_trades()[MarketEntryType.SHORT]
                self.closing_trade(order, short_open_trades)

            # Order is closing an open long trade
            elif(order.market_entry_type is MarketEntryType.SHORT and self.portfolio.has_open_trades(MarketEntryType.LONG)):
                long_open_trades = self.portfolio.get_open_trades()[MarketEntryType.LONG]
                self.closing_trade(order, long_open_trades)
            
            # Order is creating a new trade
            else:
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

            if(traded_quantity == trade_to_close.quantity):

                # Close a trade
                self.close_a_trade(trade_to_close, executed_order, traded_quantity)

                #  Move the closed trade from open trades to closed trades
                closed_trade = open_trades.pop()
                self.portfolio.add_closed_trade(closed_trade)

            elif(traded_quantity == order_quantity_to_close):
                splitted_closed_trade = self.partial_close_a_trade(trade_to_close, executed_order, traded_quantity)
                self.portfolio.add_closed_trade(splitted_closed_trade)

            order_quantity_to_close -= traded_quantity


    def partial_close_a_trade(
            self, 
            trade_to_partial_close : Trade,
            executed_order : Order,
            traded_quantity : float
    ):
        splitted_closed_trade = copy.deepcopy(trade_to_partial_close)
        splitted_closed_trade.quantity = traded_quantity

        # Partial close a trade
        trade_to_partial_close.quantity -= traded_quantity

        # Close the splitted trade
        self.close_a_trade(splitted_closed_trade, executed_order, traded_quantity)

        return splitted_closed_trade
                

    def close_a_trade(self, trade_to_close : Trade, executed_order : Order, traded_quantity : float):
        trade_to_close.profit = traded_quantity * (executed_order.executed_price - trade_to_close.entry_price)
        trade_to_close.status = TradeStatus.CLOSED
        trade_to_close.exit_time = executed_order.executed_date_time
        trade_to_close.exit_price = executed_order.executed_price

    def calculate_quantity_to_trade(self, trading_signal : Signal, current_market_data):
        # quantity_to_trade = 0.0
        # current_market_price = current_market_data['close']
        # current_portfolio_equity_value = self.portfolio.get_equity_value(current_market_price)

        # if(trading_signal is Signal.BUY):
        #     quantity_to_trade = self.portfolio.investment_rate * current_portfolio_equity_value / current_market_price

        # # Implement a more detailed short logic with borrow fee
        # elif(trading_signal is Signal.SELL):
        #     quantity_to_trade = self.portfolio.investment_rate * current_portfolio_equity_value / current_market_price


        # return quantity_to_trade
        return 1 



                

            
            
                








   
