class Signal:
    BUY = 'BUY'
    SELL = 'SELL'
    HOLD = 'HOLD'
    TRADING_SIGNALS = [ BUY , SELL ]

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

# class OrderExecutionStatus:
#     OPEN = 'OPEN' 
#     CLOSED = 'CLOSED'
#     FAILED = 'FAILED'
#     SUCCESS = [ OPEN , CLOSED]






