from backtesting.performance.performance_base import PerformanceBase
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta

class TimeSeriesMetrics(PerformanceBase):
  def __init__(self, base: PerformanceBase):
    self.base = base
    self.trades = self.base.trades
    self.initial_capital = self.base.initial_capital
  
  def equity_curve(self):
    # Formula: Equity = Initial Capital + Sum of Daily Cumulative PnL (PnL already included fees)
    daily_pnl = defaultdict(float)
    for trade in self.trades:
      exit_date = datetime.fromisoformat(trade['exit_time'].replace("Z", "+00:00")).date()
      daily_pnl[exit_date] += trade['pnl']
    
    if not daily_pnl:
      return {}

    start_date = min(daily_pnl.keys())
    end_date = max(daily_pnl.keys())
    current_equity = self.initial_capital

    date = start_date
    equity_curve = {}
    while date <= end_date:
      current_equity += daily_pnl[date]
      equity_curve[date.isoformat()] = current_equity
      date += timedelta(days=1)
    
    return equity_curve

  def daily_returns(self):
    # Formula: Daily Return = (Current Equity - Previous Equity) / Previous Equity - 1
    equity_curve = self.equity_curve()
    
    if not equity_curve:
      return {}

    dates = sorted(equity_curve.keys())
    daily_returns = {}

    for i in range(1, len(dates)):
      prev_date = dates[i - 1]
      curr_date = dates[i]
      
      prev_equity = equity_curve[prev_date]
      curr_equity = equity_curve[curr_date]

      daily_return = (curr_equity / prev_equity) - 1
      daily_returns[curr_date] = daily_return

    return daily_returns

  def cumulative_return(self):
    # Formula: Cumulative Return = (Final Equity / Initial Capital) - 1
    equity_curve = self.equity_curve()
    if not equity_curve:
      return 0.0

    final_equity = list(equity_curve.values())[-1]
    return final_equity / self.initial_capital - 1

  def max_drawdown(self):
    # Formula: Max Drawdown = (Peak Equity - Current Equity) / Peak Equity
    equity_curve = self.equity_curve()
    if not equity_curve:
        return 0.0

    peak = -float("inf")
    max_dd = 0.0

    for date in sorted(equity_curve):
        equity = equity_curve[date]
        if equity > peak:
            peak = equity
        drawdown = (equity - peak) / peak
        max_dd = min(max_dd, drawdown)

    return max_dd

  def volatility(self, annualized=True):
    # Formula: Volatility = Standard Deviation of Daily Returns
    daily_returns = self.daily_returns()
    if not daily_returns:
      return 0.0

    returns = list(daily_returns.values())
    vol = np.std(returns)

    if annualized:
      vol *= np.sqrt(252)

    return vol

  def calculate_all(self):
    scalar_metrics = {
      'cumulative_return': self.cumulative_return(),
      'max_drawdown': self.max_drawdown(),
      'volatility': self.volatility(annualized=False),
      'annualized_volatility': self.volatility(annualized=True),
    }
    time_series_metrics = {
      'strategy': {
        'datetimes': [f"{d}T00:00:00Z" for d in list(self.equity_curve().keys())] ,
        'equity_values': list(self.equity_curve().values()),
        'daily_returns': [self.daily_returns().get(date, 0.0) for date in list(self.equity_curve().keys())]
      }
    }
    return scalar_metrics, time_series_metrics