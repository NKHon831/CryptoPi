from backtesting.performance.performance_base import PerformanceBase
import numpy as np
import pandas as pd

class TradeMetrics(PerformanceBase):
  def __init__(self, base: PerformanceBase):
    self.base = base
    self.trades = self.base.trades

  def win_rate(self):
    if not self.trades:
      return 0.0
    wins = sum(1 for t in self.trades if t["pnl"] > 0)
    return wins / len(self.trades)
  
  def loss_rate(self):
    if not self.trades:
      return 0.0
    losses = sum(1 for t in self.trades if t["pnl"] < 0)
    return losses / len(self.trades)
  
  def average_win(self):
    wins = [t["pnl"] for t in self.trades if t["pnl"] > 0]
    return np.mean(wins) if wins else 0.0
  
  def average_loss(self):
    losses = [t["pnl"] for t in self.trades if t["pnl"] < 0]
    return np.mean(losses) if losses else 0.0
  
  def holding_period_stats(self, seconds=3600):
    periods = [
      (pd.to_datetime(t["exit_time"]) - pd.to_datetime(t["entry_time"])).total_seconds() / seconds  # in hours
      for t in self.trades
    ]
    return {
      "holding_period_mean": np.mean(periods) if periods else 0.0,
      "holding_period_median": np.median(periods) if periods else 0.0,
      "holding_period_max": np.max(periods) if periods else 0.0,
      "holding_period_min": np.min(periods) if periods else 0.0,
    }
  
  def profit_factor(self):
    total_win = sum(t["pnl"] for t in self.trades if t["pnl"] > 0)
    total_loss = abs(sum(t["pnl"] for t in self.trades if t["pnl"] < 0))
    return total_win / total_loss if total_loss != 0 else np.inf

  def calculate_all(self):
    return {
      "total_trades": len(self.trades),
      "win_rate": self.win_rate(),
      "loss_rate": self.loss_rate(),
      "average_win": self.average_win(),
      "average_loss": self.average_loss(),
      **self.holding_period_stats(), # Unpack the holding period stats into the top level
      "profit_factor": self.profit_factor()
    }