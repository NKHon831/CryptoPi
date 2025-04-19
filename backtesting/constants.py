class Signal:
    BUY = 'BUY'
    SELL = 'SELL'
    HOLD = 'HOLD'
    TRADING_SIGNALS = [ BUY , SELL ]
    SIGNALS = {
        BUY : 1,
        SELL : -1,
        HOLD : 0
    }

    @staticmethod
    def map_to_binary(trading_signal):
        return Signal.SIGNALS.get(trading_signal)

class ExecutionType:
    OPEN = 'OPEN'
    CLOSE = 'CLOSE'
    HIGH = 'HIGH'
    LOW = 'LOW'

class ExecutionInterval:
    MINUTE = 'MINUTE'
    HOUR = 'HOUR'
    DAY = 'DAY'
    WEEK = 'WEEK'

class OrderStatus:
    PENDING = 'PENDING'
    EXECUTED = 'EXECUTED'
    CANCELLED = 'CANCELLED'

class TradeStatus:
    OPEN = 'OPEN'
    CLOSED = 'CLOSED'
 
class MarketEntryType:
    LONG = 'LONG'
    SHORT = 'SHORT'

# class OrderExecutionStatus:
#     OPEN = 'OPEN' 
#     CLOSED = 'CLOSED'
#     FAILED = 'FAILED'
#     SUCCESS = [ OPEN , CLOSED]






