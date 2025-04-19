class PerformanceBase:
  closed_trades = None

  def __init__(self, trades, initial_capital, fee_rate=0.06):
    self.trades = trades
    self.initial_capital = initial_capital
    # excluded because already included in portfolio module
    self.append_fee(fee_rate)
    self.append_pnl()
    print(self.trades)
    print()

  def append_fee(self, rate, type="both"):
    # assume fee charged on both entry and exit

    for trade in self.trades:
      entry_price = trade["entry_price"]
      exit_price = trade["exit_price"]
      quantity = trade["quantity"]

      if type == "entry":
        fee = entry_price * quantity * (rate / 100)
      elif type == "exit":
        fee = exit_price * quantity * (rate / 100)
      elif type == "both":
        fee = (entry_price + exit_price) * quantity * (rate / 100)
      else:
        raise ValueError("Invalid fee type. Must be 'entry', 'exit', or 'both'.")

      trade["fee"] = fee

  def append_pnl(self):
    # included fee in the calculation of pnl
    for trade in self.trades:
      entry_price = trade["entry_price"]
      exit_price = trade["exit_price"]
      quantity = trade["quantity"]
      direction = trade["direction"]
      fee = trade.get("fee", 0)

      if direction == "long":
        raw_pnl = (exit_price - entry_price) * quantity
      elif direction == "short":
        raw_pnl = (entry_price - exit_price) * quantity
      else:
        raise ValueError("Invalid trade direction {direction}. Must be 'long' or 'short'.")
      
      pnl = raw_pnl - fee
      trade["pnl"] = pnl

  @staticmethod
  def import_closed_trades(closed_trades):
    PerformanceBase.closed_trades = closed_trades